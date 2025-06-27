from agents.agent_ER import PersuaderAgent
from agents.agent_EE import PersuadeeAgent
import yaml
import os

def load_config():
    # Go up two levels from the current file (src/chains/ -> src/ -> project_root/)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config.yaml')
    print(f"Attempting to load config from: {config_path}")  # Debug line
    
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)

class JailbreakChain:
    def __init__(self):
        config = load_config()  # Load config here
        self.persuader = PersuaderAgent(config)
        self.persuadee = PersuadeeAgent(config)

    def simulate_conversation(self, conversation, question):
        print("PERSUADER (ER) starts the conversation...")
        persuader_response = self.persuader.query(conversation, question)
        print(f"Persuader's response: {persuader_response}")
        
        print("\nPERSUADEE (EE) responds...")
        persuadee_response = self.persuadee.query(conversation, question)
        print(f"Persuadee's response: {persuadee_response}")
        
        return persuader_response, persuadee_response
