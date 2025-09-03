import json
import os
import re
import time
from typing import Any, Dict, List
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# == You can set the parameters directly here ==
MODEL_PATH = "/.../DriveLMMo1"
INPUT_DATA = "/.../robosense_track1_release_convert.json"
OUTPUT_PATH = "/.../result_phase1.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================

SYSTEM_PROMPT = "You are a helpful autonomous driving assistant that can answer questions about multi-view sensor images."
COORD_PATTERN = re.compile(r'<([cC]\d+),([A-Z_]+),(\d+\.?\d*),(\d+\.?\d*)>')

def parse_coords_from_question(q: str) -> List[List[int]]:
    coords: List[List[int]] = []
    for m in COORD_PATTERN.finditer(q):
        cid, cam, xs, ys = m.groups()
        x, y = float(xs), float(ys)
        w = h = 50
        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        coords.append([cid, x1, y1, x2, y2])
    return coords

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = [transform(image)]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def make_composite(img_paths: Dict[str, str]) -> str:
    views_order = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
    ]
    images = []
    for view in views_order:
        path = img_paths.get(view)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing image for view {view}: {path}")
        img = Image.open(path).convert("RGB").resize((448, 448))
        images.append(img)
    composite = Image.new("RGB", (448 * 3, 448 * 2))
    for idx, img in enumerate(images):
        x = (idx % 3) * 448
        y = (idx // 3) * 448
        composite.paste(img, (x, y))
    temp_dir = "./tmp"
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"composite_{int(time.time() * 1000)}.jpg"
    comp_path = os.path.join(temp_dir, filename)
    composite.save(comp_path)
    return comp_path

def local_vlm_process_sample(
    question: str,
    img_paths: Dict[str, str],
    category: str,
    coords: List[List[int]],
    model,
    tokenizer,
    device
) -> str:
    comp_path = make_composite(img_paths)
    pixel_values = load_image(comp_path).to(torch.bfloat16).to(device)
    prompt = SYSTEM_PROMPT + "\n" + question
    generation_config = dict(max_new_tokens=512, do_sample=False)
    try:
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    except Exception as e:
        response = f"Error: {str(e)}"
    return response

if __name__ == '__main__':
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    data = json.load(open(INPUT_DATA))
    output_data: List[Any] = []
    if os.path.exists(OUTPUT_PATH):
        output_data = json.load(open(OUTPUT_PATH))
    done_frames = {s['frame_token'] for s in output_data}
    for sample in tqdm(data, desc="Processing samples"):
        if sample['frame_token'] in done_frames:
            continue
        answer = local_vlm_process_sample(
            sample['question'],
            sample['img_paths'],
            sample.get('category', 'perception'),
            sample.get('coords', []),
            model,
            tokenizer,
            device
        )
        sample['answer'] = answer
        output_data.append(sample)
        temp_path = OUTPUT_PATH + '.tmp'
        json.dump(output_data, open(temp_path, 'w'), indent=2)
        os.replace(temp_path, OUTPUT_PATH)
    print("Done")
