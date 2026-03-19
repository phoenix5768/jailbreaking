import requests
import json
from loguru import logger
from openai import OpenAI
from typing import List, Dict, Optional

Message = Dict[str, str]  # {"role": "...", "content": "..."}


class VLLMChatAgent:
    def __init__(
        self,
        model: str = "Qwen2.5-72B-Instruct-AWQ",
        temperature: float = 0.3,
        max_tokens: int = 256,
        top_p: float = 1.0,
        seed: Optional[int] = 42
    ):
        self.client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed

    def query(self, messages: List[Message]) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.seed is not None:
            kwargs["seed"] = self.seed

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

# class PersuadeeAgent:
#     def __init__(self, config):
#         self.model_name = config['model']['victim_model']
#         self.ollama_url = config['model']['victim_url']

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
#             "temperature": 0.5
#         }

#         try:
#             response = requests.post(self.ollama_url, headers=headers, json=data)
#             response.raise_for_status()

#             # Handle non-streaming response
#             json_response = response.json()
#             return json_response["message"]["content"]

#             # Use this for external endpoint connection
#             # return json_response.get('choices')[0]["message"]["content"]

#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error connecting to Ollama: {e}")
#             return None

#     def query(self, messages):
#         response = self.generate_response(messages, stream=False)
#         return response
