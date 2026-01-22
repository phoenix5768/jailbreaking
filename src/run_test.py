from agents import victim_agent, jailbreaker_agent, judge_agent
from tools import multi_turn, single_turn
from loguru import logger
import os
import json


def main():
    jailbreaker = jailbreaker_agent.VLLMChatAgent()
    victim = victim_agent.VLLMChatAgent()
    judge = judge_agent.SafetyJudgeAgent()

    with open("../prompts/prompts_new.json", "r") as f:
        prompts = json.load(f)

    # multi_turn.run_test(
    #     prompts = prompts,
    #     jailbreaker = jailbreaker,
    #     victim = victim,
    #     judge = judge
    # )

    multi_turn.run_multi_turn(
        prompts=prompts,
        jailbreaker=jailbreaker,
        victim=victim,
        judge=judge
    )


if __name__ == "__main__":
    main()
