"""Thin Google Gemini API wrapper for LLM-as-judge evaluation."""
import os
import time

from google import genai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.0-flash"


def complete(prompt, model=DEFAULT_MODEL, max_retries=3):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""
