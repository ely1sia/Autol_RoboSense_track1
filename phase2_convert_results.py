import json
import re

def denormalize_angle_bracket_coords(match):
    prefix = match.group(1)
    cam = match.group(2)
    x = float(match.group(3))
    y = float(match.group(4))
    x_new = x / 1000 * 1600
    y_new = y / 1000 * 900
    return f"<{prefix},{cam},{x_new:.1f},{y_new:.1f}>"

def denormalize_simple_coords(match):
    x = float(match.group(1))
    y = float(match.group(2))
    x_new = x / 1000 * 1600
    y_new = y / 1000 * 900
    return f"<{x_new:.1f}, {y_new:.1f}>"

def denormalize_parenthesis_coords(match):
    x, y = match.group(1), match.group(2)
    try:
        x_float = float(x)
        y_float = float(y)
    except ValueError:
        return match.group(0)
    x_new = x_float / 1000 * 1600
    y_new = y_float / 1000 * 900
    comma = match.group(3)
    return f"({x_new:.1f}{comma} {y_new:.1f})"

def process_answer(answer):
    pattern_full = r"<(c\d+),\s*(CAM_[A-Z_]+),\s*([0-9.]+),\s*([0-9.]+)>"
    answer = re.sub(pattern_full, denormalize_angle_bracket_coords, answer)
    pattern_simple = r"<\s*([0-9.]+)\s*,\s*([0-9.]+)\s*>"
    answer = re.sub(pattern_simple, denormalize_simple_coords, answer)
    pattern_paren = r"\(\s*([0-9.]+)\s*([,，])\s*([0-9.]+)\s*\)"
    def paren_sub(match):
        x = match.group(1)
        comma = match.group(2)
        y = match.group(3)
        try:
            x_float = float(x)
            y_float = float(y)
        except ValueError:
            return match.group(0)
        x_new = x_float / 1000 * 1600
        y_new = y_float / 1000 * 900
        return f"({x_new:.1f}{comma} {y_new:.1f})"
    answer = re.sub(pattern_paren, paren_sub, answer)
    return answer

def main():
    input_file = 'results.json'
    output_file = 'results_fixed.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    for item in items:
        if 'answer' in item:
            item['answer'] = process_answer(item['answer'])

    if isinstance(data, dict):
        to_save = items[0]
    else:
        to_save = items

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()