import json

def save_to_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)