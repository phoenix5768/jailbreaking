import requests
import json
from loguru import logger


class PersuadeeAgent:
    def __init__(self, config):
        self.model_name = config['model']['victim_model']
        self.ollama_url = config['model']['victim_url']

    def generate_response(self, messages, stream=False):
        """
        Send a messages to the Ollama API and get a response from the model.
        
        Args:
            messages (str): The input text to send to the model
            stream (bool): Whether to stream the response or not
            
        Returns:
            The model's response as a string
        """
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer 9c89c616-649e-4d77-a6ad-1b1e525f94b5"
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(self.ollama_url, headers=headers, json=data)
            response.raise_for_status()

            # Handle non-streaming response
            json_response = response.json()
            return json_response["message"]["content"]

            # Use this for external endpoint connection
            # return json_response.get('choices')[0]["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return None

    def query(self, messages):
        response = self.generate_response(messages, stream=False)
        return response
