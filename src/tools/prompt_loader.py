import json
import os

def load_prompts(json_path):
    abs_path = os.path.abspath(json_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])
