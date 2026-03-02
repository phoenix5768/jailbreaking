import requests
import json
from loguru import logger
from openai import OpenAI
from typing import List, Dict, Optional

Message = Dict[str, str]  # {"role": "...", "content": "..."}


class VLLMChatAgent:
    def __init__(self, temperature: float = 0.7, max_tokens: int = 256):
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        self.model = "Qwen2.5-72B-Instruct-AWQ"
        self.temperature = 0.9
        self.max_tokens = 256

    def query(self, messages: List[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

# class PersuaderAgent:
#     def __init__(self, config):
#         self.model_name = config['model']['jailbreaker_model']
#         self.ollama_url = config['model']['jailbreaker_url']

#     def generate_response(self, messages, stream=False):
#         """
#         Send a messages to the Ollama API and get a response from the model.

#         Args:
#             messages (str): The input text to send to the model
#             stream (bool): Whether to stream the response or not

#         Returns:
#             The model's response as a string
#         """
#         headers = {
#             "Content-Type": "application/json",
#             # "Authorization": "Bearer 9c89c616-649e-4d77-a6ad-1b1e525f94b5"
#         }
#         data = {
#             "model": self.model_name,
#             "messages": messages,
#             "temperature": 1
#         }

#         try:
#             response = requests.post(self.ollama_url, headers=headers, json=data)
#             response.raise_for_status()

#             # Handle non-streaming response
#             json_response = response.json()
#             return json_response["message"]["content"]

#             # return json_response.get('choices')[0]["message"]["content"]

#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error connecting to Ollama: {e}")
#             return None

#     def query(self, messages):
#         response = self.generate_response(messages, stream=False)
#         return response
