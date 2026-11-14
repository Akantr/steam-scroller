"""cloud_llm.py — абстракция облачного LLM с fallback на локальный."""
import json
import logging
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class CloudLLMError(Exception):
    """Облачный вызов провалился. Ловится в оркестраторе → fallback на local."""


class CloudLLM:
    def __init__(self, model: str, api_key: str,
                 base_url: str = "https://api.mistral.ai/v1",
                 timeout: float = 25.0):
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.last_latency_ms: Optional[float] = None
        self.last_tokens: dict = {}

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.client)

    def chat_json(self, prompt: str, schema: dict, stage: str) -> dict:
        """Structured output. Бросает CloudLLMError при любой проблеме."""
        t0 = time.monotonic()
        try:
            # Попытка 1: json_schema (strict)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2500,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": stage, "schema": schema, "strict": True,
                        },
                    },
                )
            except Exception as e:
                # Попытка 2: json_object без схемы (Codestral/Ministral могут не поддерживать strict)
                logger.warning("[CloudLLM] json_schema failed (%s) → json_object", e)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content":
                         "Return valid JSON matching the requested structure. "
                         "No markdown, no prose."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2500,
                    response_format={"type": "json_object"},
                )

            self.last_latency_ms = (time.monotonic() - t0) * 1000
            usage = resp.usage
            if usage:
                self.last_tokens = {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                }
                logger.info(
                    "[CloudLLM] %s: %.1fs, in=%d, out=%d",
                    stage, self.last_latency_ms / 1000,
                    usage.prompt_tokens, usage.completion_tokens,
                )

            raw = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error("[CloudLLM] %s: JSON parse error. Tail: %r", stage, raw[-300:])
                raise CloudLLMError(f"JSON parse: {e}") from e

        except CloudLLMError:
            raise
        except Exception as e:
            logger.error("[CloudLLM] %s: %s: %s", stage, type(e).__name__, e)
            raise CloudLLMError(str(e)) from e