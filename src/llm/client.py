"""
LLM client abstraction with caching, cost tracking, and provider adapters.
Author: Chaima Yedes
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm

logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """Tracks cumulative token usage and estimated cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    call_count: int = 0

    def record(self, prompt_tok: int, completion_tok: int, cost: float) -> None:
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += prompt_tok + completion_tok
        self.estimated_cost_usd += cost
        self.call_count += 1


@dataclass
class LLMResponse:
    """Wrapper around a single LLM completion."""

    content: str
    parsed: Optional[dict[str, Any]] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class LLMClient(ABC):
    """Abstract LLM client with caching and cost tracking."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.usage = LLMUsage()
        self._cache: dict[str, LLMResponse] = {}

    def _cache_key(self, prompt: str) -> str:
        raw = f"{self.model}::{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @abstractmethod
    def _call(
        self, prompt: str, schema: dict[str, Any] | None, attachments: list[Any] | None
    ) -> LLMResponse:
        ...

    def complete(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Send a prompt to the LLM and return parsed JSON output.

        Parameters
        ----------
        prompt : str
            The user prompt.
        schema : dict, optional
            JSON schema for structured output.
        attachments : list, optional
            Images or other multimodal content.
        use_cache : bool
            Whether to use the prompt cache.
        """
        key = self._cache_key(prompt)
        if use_cache and key in self._cache:
            logger.debug("Cache hit for prompt hash %s", key[:12])
            resp = self._cache[key]
            resp.cached = True
            return resp.parsed or {"text": resp.content}

        resp = self._call(prompt, schema, attachments)
        self._cache[key] = resp
        return resp.parsed or {"text": resp.content}


class LiteLLMAdapter(LLMClient):
    """
    Concrete adapter that delegates to litellm for provider-agnostic
    access to language models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self.api_base = api_base

    def _call(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        attachments: list[Any] | None,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        system_msg = (
            "You are a data quality analysis assistant. "
            "Always respond with valid JSON when a schema is provided."
        )
        messages.append({"role": "system", "content": system_msg})

        user_content: list[dict[str, Any]] | str
        if attachments:
            user_content = [{"type": "text", "text": prompt}]
            for att in attachments:
                if isinstance(att, str) and att.startswith("data:image"):
                    user_content.append({"type": "image_url", "image_url": {"url": att}})
                elif isinstance(att, str):
                    user_content.append({"type": "text", "text": att})
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        if schema:
            schema_instruction = (
                f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
            if isinstance(user_content, str):
                messages[-1]["content"] += schema_instruction
            else:
                user_content.append({"type": "text", "text": schema_instruction})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        t0 = time.perf_counter()
        response = litellm.completion(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        prompt_tok = usage.prompt_tokens if usage else 0
        completion_tok = usage.completion_tokens if usage else 0
        cost = litellm.completion_cost(completion_response=response) or 0.0

        self.usage.record(prompt_tok, completion_tok, cost)

        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            clean = content.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                pass

        return LLMResponse(
            content=content,
            parsed=parsed,
            model=self.model,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            latency_ms=round(latency, 2),
            cached=False,
        )


class ClaudeAdapter(LiteLLMAdapter):
    """Preconfigured adapter for Anthropic models via litellm."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)


class OpenAIAdapter(LiteLLMAdapter):
    """Preconfigured adapter for OpenAI models via litellm."""

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
