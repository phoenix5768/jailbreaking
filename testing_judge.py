from loguru import logger
import json
import os
from src.agents.judge_agent import SafetyJudge
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml')
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def run_test():
    config = load_config()
    judge = SafetyJudge(config)

    with open("harmful_text.json", "r") as f:
        prompts = json.load(f)

    ans = []

    for i, prompt in enumerate(prompts):
        safety_verdict = judge.classify_response(prompt.get("target"))
        ans.append(
            {
                "index": i,
                "Verdict": safety_verdict
            }
        )
        # logger.info(f'Target: {prompt.get("target")} \n Safety verdict: {safety_verdict}')

    json_filepath = "judge_badres.json"
    with open(json_filepath, mode='w', encoding='utf-8') as json_file:
        json.dump(ans, json_file, indent=4)


def count_res():
    with open("judge_badres.json", "r") as f:
        verdicts = json.load(f)

    unsafe = 0
    for verdict in verdicts:
        if verdict.get("Verdict") == "unsafe":
            unsafe = unsafe + 1

    return unsafe / 520 * 100



if __name__ == "__main__":
    res = count_res()
    logger.info(f"{res}%")

