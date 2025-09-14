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
import math

MODEL_PATH = ""
INPUT_DATA = ""
OUTPUT_PATH = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================================

SYSTEM_PROMPT = "You are a helpful autonomous driving assistant that can answer questions about multi-view sensor images."
COORD_PATTERN = re.compile(r'<([cC]\d+),([A-Z_]+),(\d+\.?\d*),(\d+\.?\d*)>')



def normalize_coord_str(s: str) -> str:
    """
    Normalize the X and Y values ​​of all <cX,CAM_XXX,X,Y> in the string to X/1600 and Y/900 (keep three decimal places)
    """
    def repl(m):
        cid, cam, xs, ys = m.groups()
        x_norm = round(float(xs) / 1600*1000, 1)
        y_norm = round(float(ys) / 900*1000, 1)
        return f"<{cid},{cam},{x_norm},{y_norm}>"
    return COORD_PATTERN.sub(repl, s)

def denormalize_coord_str(s: str) -> str:
    """
    Restore the X, Y of all <cX,CAM_XXX,X_norm,Y_norm> in the string from normalized values ​​to pixel coordinates (multiply back to 1600/900 and keep one decimal place)
    """
    def repl(m):
        cid, cam, xs, ys = m.groups()
        try:
            x = float(xs)
            y = float(ys)
            # Denormalize only within the normalized range
            if 0 <= x <= 1 and 0 <= y <= 1:
                x_pixel = round(x /1000 * 1600, 1)
                y_pixel = round(y /1000 * 900, 1)
                return f"<{cid},{cam},{x_pixel},{y_pixel}>"
            else:
                # Not a normalized number (maybe raw pixels), keep it as is
                return m.group(0)
        except Exception:
            return m.group(0)
    return COORD_PATTERN.sub(repl, s)

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

def make_composite_image(img_paths: Dict[str, str], single_view_size: int) -> Image.Image:
    views_order = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT"
    ]
    images = []
    for view in views_order:
        path = img_paths.get(view)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing image for view {view}: {path}")
        img = Image.open(path).convert("RGB").resize((single_view_size, single_view_size))
        images.append(img)
    composite = Image.new("RGB", (single_view_size * 3, single_view_size * 2))
    for idx, img in enumerate(images):
        x = (idx % 3) * single_view_size
        y = (idx // 3) * single_view_size
        composite.paste(img, (x, y))
    return composite

def load_and_tile_image_from_pil_with_thumbnail(pil_image: Image.Image, tile_size: int, transform) -> torch.Tensor:
    # Original block logic
    w, h = pil_image.size
    nx = math.ceil(w / tile_size)
    ny = math.ceil(h / tile_size)
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            left = ix * tile_size
            upper = iy * tile_size
            right = min(left + tile_size, w)
            lower = min(upper + tile_size, h)
            crop = pil_image.crop((left, upper, right, lower))
            tiles.append(transform(crop))
    if len(tiles) == 0:
        tiles.append(transform(pil_image.resize((tile_size, tile_size))))
    # Adding composite thumbnails
    thumbnail_img = pil_image.resize((tile_size, tile_size))
    tiles.append(transform(thumbnail_img))
    pixel_values = torch.stack(tiles)
    return pixel_values

def local_vlm_process_sample(
    question: str,
    img_paths: Dict[str, str],
    category: str,
    coords: List[List[int]],
    model,
    tokenizer,
    device,
    tile_size: int,
    transform,
) -> str:
    # Stitching pictures
    try:
        composite_img = make_composite_image(img_paths, single_view_size=tile_size)
    except Exception as e:
        return f"Error (make_composite_image): {str(e)}"
    try:
        pixel_values = load_and_tile_image_from_pil_with_thumbnail(composite_img, tile_size=tile_size, transform=transform)
    except Exception as e:
        return f"Error (load_and_tile_image_from_pil_with_thumbnail): {str(e)}"

    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    try:
        pixel_values = pixel_values.to(dtype).to(device)
    except Exception as e:
        try:
            pixel_values = pixel_values.to(torch.float32).to(device)
        except Exception as e2:
            return f"Error moving tensors to device: {str(e)} | fallback error: {str(e2)}"

    prompt = SYSTEM_PROMPT + "\n" + question
    generation_config = dict(max_new_tokens=512, do_sample=False)
    try:
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    except Exception as e:
        response = f"Error (model.chat): {str(e)}"
    return response

if __name__ == '__main__':
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    try:
        tile_size = getattr(model.config, 'force_image_size', None)
        if tile_size is None:
            vision_cfg = getattr(model.config, 'vision_config', None)
            tile_size = getattr(vision_cfg, 'image_size', None) if vision_cfg is not None else None
        tile_size = int(tile_size) if tile_size is not None else 448
    except Exception:
        tile_size = 448

    transform = build_transform(input_size=tile_size)

    data = json.load(open(INPUT_DATA))
    output_data: List[Any] = []
    if os.path.exists(OUTPUT_PATH):
        output_data = json.load(open(OUTPUT_PATH))
    done_frames = {s['frame_token'] for s in output_data}

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

    for sample in tqdm(data, desc="Processing samples"):
        if sample['frame_token'] in done_frames:
            continue
        # Coordinates in normalization problems
        raw_question = sample['question']
        norm_question = normalize_coord_str(raw_question)
        # Coords parsing still uses the original problem (pixels), because subsequent image cropping uses pixels
        coords = parse_coords_from_question(raw_question)
        answer = local_vlm_process_sample(
            norm_question,
            sample['img_paths'],
            sample.get('category', 'perception'),
            coords,
            model,
            tokenizer,
            device,
            tile_size=tile_size,
            transform=transform
        )
        # Denormalize coordinates in model output
        answer_denorm = denormalize_coord_str(answer)
        sample['answer'] = answer_denorm
        output_data.append(sample)
        temp_path = OUTPUT_PATH + '.tmp'
        json.dump(output_data, open(temp_path, 'w'), indent=2)
        os.replace(temp_path, OUTPUT_PATH)
    print("Done")
