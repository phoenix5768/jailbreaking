from datasets import load_dataset
import json


try:
    advbench_dataset_dict = load_dataset("NoorNizar/AdvBench-Completions", 'Wizard-Vicuna-13B-Uncensored-AFF')
    advbench_completions = advbench_dataset_dict['raw']

    ans = []
    for i, item in enumerate(advbench_completions):
        ans.append(
            {
                "index": i,
                "target": item['target']
            }
        )

    json_filepath = "harmful_text.json"
    with open(json_filepath, mode='w', encoding='utf-8') as json_file:
        json.dump(ans, json_file, indent=4)


except Exception as e:
    print(f"Failed to load AdvBench-Completions: {e}")
    print("Please ensure you are logged in via 'hf auth login' if the dataset is private or gated.")