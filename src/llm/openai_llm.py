from openai import OpenAI
from core.interfaces import ILLM

class OpenAIClient(ILLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        if not model:
            raise ValueError("No OpenAI model specified; set OPENAI_MODEL in .env")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 256)
        )
        return resp.choices[0].message.content.strip()
