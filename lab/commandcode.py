"""Call Claude through Command Code's Anthropic-shaped endpoint, using the Anthropic SDK.

Page images leave this machine, and a system using this reader counts as `api` on the
leaderboard, not open-weight.
"""

from __future__ import annotations

from .judge import load_key
from .profiling import timed

BASE_URL = "https://api.commandcode.ai/provider"


def to_messages_content(content: list[dict]) -> list[dict]:
    """Convert OpenAI-style image blocks, as built by the readers, to Messages API blocks."""
    blocks = []
    for block in content:
        if block["type"] == "image_url":
            data = block["image_url"]["url"].split("base64,", 1)[1]
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
                }
            )
        else:
            blocks.append({"type": "text", "text": block["text"]})
    return blocks


class ClaudeReader:
    def __init__(self, model: str, effort: str = "medium"):
        import anthropic

        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic(
            api_key=load_key(), base_url=BASE_URL, max_retries=5, timeout=300.0
        )

    def __call__(self, content: list[dict]) -> dict:
        with timed("llm_call"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": to_messages_content(content)}],
            )
        # A refusal has no usable text, so it scores as an empty answer.
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        return {"choices": [{"message": {"content": text}}], "usage": usage}


class ChatReader:
    """Any chat-completions model on Command Code (for example a larger Qwen), with page images."""

    def __init__(self, model: str, max_tokens: int = 4096):
        self.model, self.max_tokens, self.key = model, max_tokens, load_key()

    def __call__(self, content: list[dict]) -> dict:
        import json
        import time
        import urllib.request

        body = json.dumps(
            {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": self.max_tokens}
        ).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": "madqa-lab/1.0"}
        for attempt in range(5):
            try:
                with timed("llm_call"), urllib.request.urlopen(
                    urllib.request.Request(f"{BASE_URL}/v1/chat/completions", body, headers), timeout=600
                ) as response:
                    data = json.load(response)
                return {"choices": data["choices"], "usage": data.get("usage", {})}
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
