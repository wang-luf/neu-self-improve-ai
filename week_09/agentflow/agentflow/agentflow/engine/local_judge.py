"""Local LLM judge (assignment addition).

Drop-in replacement for the gpt-4o scorer used by train/utils.py (training
reward) and test/calculate_score_unified.py (evaluation). It sends the same
judge prompt to a local vLLM OpenAI-compatible server and parses the reply into
the caller's pydantic `response_format` with vLLM's structured output.
Enabled by setting JUDGE_BASE_URL (e.g. http://localhost:8000/v1).
"""

import os

from openai import OpenAI


class LocalJudge:
    def __init__(self, model_string=None, base_url=None):
        self.model_string = model_string or os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.client = OpenAI(base_url=base_url or os.environ["JUDGE_BASE_URL"], api_key="EMPTY")

    def __call__(self, prompt, response_format=None, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        if response_format is None:
            reply = self.client.chat.completions.create(model=self.model_string, messages=messages, temperature=0)
            return reply.choices[0].message.content
        reply = self.client.beta.chat.completions.parse(
            model=self.model_string, messages=messages, temperature=0, response_format=response_format
        )
        return reply.choices[0].message.parsed
