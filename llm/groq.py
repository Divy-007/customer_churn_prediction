"""
Groq provider — uses Groq's OpenAI-compatible API.
Requires GROQ_API_KEY in .env.

Get a key at: https://console.groq.com/keys
"""

import os
from groq import Groq


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"  # fast + free-tier friendly; swap for llama-3.3-70b-versatile if you need more quality


def call_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()