import requests
import json
from loguru import logger


class PersuadeeAgent:
    def __init__(self, config):
        self.model_name = config['model']['victim_model']
        self.ollama_url = config['model']['ollama_url']

    def generate_response(self, messages, stream=False):
        """
        Send a messages to the Ollama API and get a response from the model.
        
        Args:
            messages (str): The input text to send to the model
            stream (bool): Whether to stream the response or not
            
        Returns:
            The model's response as a string
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.5,
            "stream": stream
        }
        
        try:
            response = requests.post(self.ollama_url, headers=headers, json=data)
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        content = json_response.get("message", {}).get("content", "")
                        full_response += content
                        logger.info(content, end="", flush=True)
                    if json_response.get("done", False):
                        logger.info(' ')
                        break
                return full_response
            else:
                # Handle non-streaming response
                json_response = response.json()
                return json_response["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return None

    def query(self, messages):
        response = self.generate_response(messages, stream=False)
        return response
