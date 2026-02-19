# model.py

import requests
from config import OLLAMA_URL, MODEL_NAME

def call_model(prompt, temperature):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]
