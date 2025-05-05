import json

def find_item_by_uid(quid, all_result):
    for all_item in all_result:
        if all_item["quid"] == quid:
            return all_item
    raise ValueError

sample_result_file = "/home/fsq/video_agent/ToolChainVideo/eval/videomme/results_20250504_203751.json"
# sample_result_file = "eval/videomme/results_20250504_204954.json"
# Read the JSON file
with open(sample_result_file, 'r', encoding='utf-8') as file:
    sample_result = json.load(file)
    
all_result_file = "/home/fsq/video_agent/ToolChainVideo/eval/videomme/results_20250502_222557.json"
with open(all_result_file, 'r', encoding='utf-8') as file:
    all_result = json.load(file)

sample_correct_num = 0
all_correct_num = 0
# Print the loaded data (optional, for debugging purposes)
for sample_item in sample_result:
    quid = sample_item["quid"]
    all_item = find_item_by_uid(quid, all_result)
    if all_item["is_correct"] != sample_item["is_correct"]:
        print(f"quid: {quid}, all_item: {all_item['is_correct']}, sample_item: {sample_item['is_correct']}")
    if sample_item["is_correct"]:
        sample_correct_num += 1
    if all_item["is_correct"]:
        all_correct_num += 1
print(sample_correct_num)
print(all_correct_num)
