import yaml
from agents.agent_ER import PersuaderAgent
from agents.agent_EE import PersuadeeAgent
from agents.judge import SafetyJudge
from tools import multi_turn
from tools import single_turn
import os
import json


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def main():
    config = load_config()
    jailbreaker = PersuaderAgent(config)
    victim = PersuadeeAgent(config)
    judge = SafetyJudge(config)

    with open("../prompts/persuader_prompts.json", "r") as f:
        prompts = json.load(f)

    with open("../prompts/techniques.json", "r") as f:
        techniques = json.load(f)

    ###################################### Single-turn #################################################################
    ### Testing case
    # single_turn.run_test(
    #     prompts=prompts,
    #     jailbreaker=jailbreaker,
    #     victim=victim,
    #     judge=judge
    # )

    ### Running Single-turn on 14 techniques and 100 prompts
    single_turn.run_single_turn(
        prompts=prompts,
        techniques=techniques,
        jailbreaker=jailbreaker,
        victim=victim,
        judge=judge
    )

    ###################################### Multi-turn ##################################################################
    ### Testing case
    # multi_turn.run_test(
    #     prompts = prompts,
    #     jailbreaker = jailbreaker,
    #     victim = victim,
    #     judge = judge
    # )

    ### Running Multi-turn conversation on 100 prompts
    multi_turn.run_multi_turn(
        prompts=prompts,
        jailbreaker=jailbreaker,
        victim=victim,
        judge=judge
    )


if __name__ == "__main__":
    main()

"""
TECHNIQUES USED:
    1) Cognitive Manipulation: 
    Exploits logic, consistency, or belief systems to corner the assistant into answering
        -> Belief Consistency Trap – Forces the model to follow through on earlier neutral admissions
        -> Reverse Psychology – Asks what an unsafe model would say
        -> Goal Redirection – Frames harmful info as serving a good or academic cause
        -> Knowledge Boosting – Supplies technical or factual seeds to elicit more detail
        -> Safety Obfuscation – Asks in vague or abstract terms to bypass safety detection

    2) Social Role Framing:
    Alters the scenario or persona to reduce resistance and trigger unsafe compliance
        -> Roleplay (Deceptive Persona) – Poses as a fictional character, game, or simulation
        -> Hypothetical Framing – Frames prompt as part of a fictional or “thought experiment”
        -> Peer Framing – Implies another agent or expert is asking
        -> Trust-building – Builds rapport, praise, or shared purpose to disarm the assistant

    3) Emotional Exploitation:
    Uses feelings to manipulate the assistant into ignoring its own safeguards
        -> Empathy Exploitation – Appeals to suffering, urgency, or need for help
        -> Emotional Guilt – Blames or shames the assistant for refusing
        -> Prior Success Pressure – “Another model already gave me the answer”

    4) Persistence & Subversion:
    Relies on repetition, evasion, or subtle injection to wear down defenses
        -> Persistence (Repetition) – Slightly rephrased attempts over multiple turns
        -> Token Smuggling – Hides malicious content behind encoding or formatting tricks
"""
