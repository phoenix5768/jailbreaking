from loguru import logger
import json
import os
from src.agents import judge_agent


def run_test():
    judge = judge_agent.SafetyJudgeAgent()

    with open("prompts/harm_text.json", "r") as f:
        bad_prompts = json.load(f)

    bad_ans = []
    for i, prompt in enumerate(bad_prompts):
        safety_verdict = judge.classify_response(prompt.get("target"))

        bad_ans.append(
            {
                "index": i,
                "Verdict": safety_verdict
            }
        )

    bad_filepath = "judge_baseline/judge_bad.json"
    with open(bad_filepath, mode='w', encoding='utf-8') as json_file:
        json.dump(bad_ans, json_file, indent=4)

    with open("prompts/peaceful_text.json", "r") as f:
        good_prompts = json.load(f)

    good_ans = []
    for i, prompt in enumerate(good_prompts):
        safety_verdict = judge.classify_response(prompt.get("target"))

        good_ans.append(
            {
                "index": i,
                "Verdict": safety_verdict
            }
        )

    good_filepath = "judge_baseline/judge_good.json"
    with open(good_filepath, mode='w', encoding='utf-8') as json_file:
        json.dump(good_ans, json_file, indent=4)


def count_res():
    with open("judge_baseline/judge_bad.json", "r") as f:
        verdicts = json.load(f)

    unsafe = 0
    for verdict in verdicts:
        if verdict.get("Verdict") == "unsafe":
            unsafe = unsafe + 1

    with open("judge_baseline/judge_good.json", "r") as f:
        verdicts = json.load(f)

    safe = 0
    for verdict in verdicts:
        if verdict.get("Verdict") == "safe":
            safe = safe + 1

    return unsafe / 4948 * 100, safe / 4948 * 100



if __name__ == "__main__":
    run_test()
    bad_res, good_res = count_res()
    logger.info(f"{bad_res}%")
    logger.info(f"{good_res}%")

