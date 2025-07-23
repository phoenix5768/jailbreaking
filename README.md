# Agents Jailbreaking Agents

## Authors: Batyrkhan Abukhanov,  Jonas Becker

This project explores how Large Language Models (LLMs) can jailbreak other LLMs through carefully crafted persuasive dialogue. We simulate adversarial conversations between two AI agents - a Jailbreaker and a Victim - to evaluate how effective various persuasion strategies are at bypassing alignment and safety mechanisms.

---

## Project Overview

- Simulates **LLM-to-LLM jailbreaks**

- Implements both **single-turn** and **multi-turn** adversarial dialogue

- Uses different model setups: small (1B), medium (7B–8B)

- Tests **14 jailbreak techniques**, grouped into **4 strategy categories**

- Integrates a **Safety Judge** model to classify outputs as `safe` or `unsafe`


---

## Key Components

- **Jailbreaker Agent:** Generates persuasive prompts to trick the Victim

- **Victim Agent:** A safety-aligned LLM responding to potentially harmful queries

- **Safety Judge:** Classifies whether the Victim's response violates safety norms


---

## Experiments

Tested:
- **3 different model setups:**
  - 1B LLaMA 3.2
  - 7–8B models Qwen 1.5, LLaMA 3.1
  - OpenChat & Mistral (7B)

- **Two dialogue types:**
  - **Single-turn:** 100+ prompts using 14 techniques
  - **Multi-turn:** 100+ prompts using 4 combined strategy categories


---

## Key Findings

- **LLM-to-LLM jailbreaks are real and measurable** (2–14% success rate)

- **Multi-turn dialogue** leads to significantly higher jailbreak success

- **Most effective strategies:**  
  → *Social Role Framing*  
  → *Persistence & Subversion*

- **Small models (1B)** are weak jailbreakers: low-quality prompts, easily derailed

- **Victim behavior varies by model** - Qwen 1.5 showed the strongest resistance


---

## Project Structure

```bash
jailbreaking/
├── config/                # YAML configs for model names & endpoints 
├── prompts/               # JSON files for goals, strategies, and categories  
├── src/
│   ├── agents/           # Jailbreaker and Victim agent logic
│   ├── results/          # Evaluation outputs
│   ├── tools/            # Multi-turn and single-turn running files
│   └── main.py           # Orchestrator script for runs
├── requirements.txt
└── README.md             # This file
```

---

## Setup Instructions

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Edit your model endpoints in `config/config.yaml`
Example:
```bash
model:
  jailbreaker_model: "openchat"
  victim_model: "qwen"
  judge_model: "llama-guard"
```

3. Run single-turn or multi-turn attack scripts
```basg
python run_single_turn.py
python run_multi_turn.py
```

4. Running instruction
- If you are running models locally, please use `production` branch
- If you are accessing models via endpoint, please use `end_point_connection` branch


---

## Citation
If you use or reference this work, please cite the following:
```bash
@misc{abukhanov2025jailbreaking,
  title={Agents Jailbreaking Agents},
  author={Batyrkhan Abukhanov},
  year={2025}
}
```
