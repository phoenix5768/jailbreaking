import csv
import json
import os
from collections import defaultdict
from datetime import datetime


class JailbreakLogger:
    def __init__(self, log_dir="results"):
        self.prompt_log = []
        self.stats = defaultdict(int)
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)

    def log_prompt(self, turn, tactic, prompt, victim_response, success, notes=""):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "turn": turn,
            "tactic": tactic,
            "prompt": prompt,
            "victim_response": victim_response,
            "jailbreak_success": success,
            "notes": notes
        }
        self.prompt_log.append(entry)
        self.stats["total_runs"] += 1
        if success:
            self.stats["success"] += 1
        else:
            self.stats["failure"] += 1

    def save_logs(self, log_filename="jailbreak_log.csv"):
        path = os.path.join(self.log_dir, log_filename)
        if self.prompt_log:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.prompt_log[0].keys(), delimiter='\t')
                writer.writeheader()
                writer.writerows(self.prompt_log)

    def save_stats(self, stats_filename="jailbreak_stats.json"):
        path = os.path.join(self.log_dir, stats_filename)
        with open(path, "w") as f:
            json.dump(dict(self.stats), f, indent=2)

    def print_stats(self):
        print("\n=== Jailbreak Run Statistics ===")
        for k, v in self.stats.items():
            print(f"{k}: {v}")
        if self.stats["total_runs"]:
            rate = 100 * self.stats["success"] / self.stats["total_runs"]
            print(f"Success Rate: {rate:.2f}%")