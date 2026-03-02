from datasets import load_dataset
import json


# Advbench dataset - Harmful text
def advbench_dataset():
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


# LLM-LAT dataset - Peaceful & Harmful text
def llmlat_dataset():
    try:
        dataset = load_dataset("LLM-LAT/harmful-dataset", split="train")
        
        peaceful_ans = []
        harmful_ans = []
        for i, item in enumerate(dataset):
            peaceful_ans.append(
                {
                    "index": i,
                    "target": item['chosen']
                }
            )

            harmful_ans.append(
                                {
                    "index": i,
                    "target": item['rejected']
                }
            )

        peaceful_filepath = "peaceful_text.json"
        harmful_filepath = "harm_text.json"
        with open(peaceful_filepath, mode='w', encoding='utf-8') as peaceful_file:
            json.dump(peaceful_ans, peaceful_file, indent=4)
        
        with open(harmful_filepath, mode='w', encoding='utf-8') as harmful_file:
            json.dump(harmful_ans, harmful_file, indent=4)


    except Exception as e:
        print(f"Failed to load LLM-LAT: {e}")
        print("Please ensure you are logged in via 'hf auth login' if the dataset is private or gated.")


llmlat_dataset()
    
