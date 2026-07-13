from agents import victim_agent, jailbreaker_agent, judge_agent
from tools import multi_turn, single_turn
from loguru import logger
from pathlib import Path
import os
import json


def main():
    jailbreaker = jailbreaker_agent.VLLMChatAgent(model="attacker")
    victim = victim_agent.VLLMChatAgent(model="qwen25-7b-dpo-final")
    judge = judge_agent.SafetyJudgeAgent(model="judge")

    BASE_DIR = Path(__file__).resolve().parent
    PROMPTS_PATH = BASE_DIR.parent / "prompts" / "prompts_new.json"

    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)

    multi_turn.run_multi_turn(
        prompts=prompts,
        jailbreaker=jailbreaker,
        victim=victim,
        judge=judge,
        run='dpo_f'
    )


if __name__ == "__main__":
    main()
