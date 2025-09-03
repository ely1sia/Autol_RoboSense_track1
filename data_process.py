import json
import re

def extract_final_answer(answer):
    """
    Extract the content after Final Answer in the answer field
    """
    match = re.search(r'\*\*Final Answer\*\*\s*:?\s*(.+)', answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer

def contains_angle_brackets(text):
    """
    Checks whether the string contains content in the form of <xxx>
    """
    return bool(re.search(r'<[^>]+>', text))

def filter_answers(input_path, output_path):
    """
    Filter and process the answer field in the results.ipynb file
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        if (
            item.get('category') == 'perception'
            and contains_angle_brackets(item.get('question', ''))
        ):
            item['answer'] = extract_final_answer(item['answer'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Processing completed, results saved to {output_path}')

filter_answers('result_phase1.json', 'results.json')