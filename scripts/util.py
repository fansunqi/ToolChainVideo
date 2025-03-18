import json

def save_to_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def parse_answer(answer):
    good_ans_list = answer.get("good_anwsers")
    if good_ans_list:
        pass
    else:
        return None