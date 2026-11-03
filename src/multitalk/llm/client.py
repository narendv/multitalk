"""Chat clients for OpenAI and Ollama models, with cost tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from ..config import LLMConfig

# USD per token as (prompt, completion).
PRICES = {
    "gpt-4-turbo": (1e-5, 3e-5),
    "gpt-4-turbo-preview": (1e-5, 3e-5),
    "gpt-3.5-turbo": (5e-7, 15e-7),
    "gpt-4o": (5e-6, 15e-6),
}


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient(Protocol):
    def complete(self, messages: list[dict]) -> Completion: ...


class OpenAIClient:
    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
            )
        self._client = OpenAI(api_key=api_key)
        self._config = config

    def complete(self, messages: list[dict]) -> Completion:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            top_p=self._config.top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return Completion(
            text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        import ollama

        self._ollama = ollama
        self._model = config.model
        ollama.show(self._model)

    def complete(self, messages: list[dict]) -> Completion:
        response = self._ollama.chat(model=self._model, messages=messages)
        return Completion(text=response["message"]["content"])


def make_client(config: LLMConfig) -> ChatClient:
    if "llama" in config.model:
        return OllamaClient(config)
    return OpenAIClient(config)


class Usage:
    """Estimated cost across every LLM agent in a run."""

    def __init__(self, config: LLMConfig) -> None:
        self._prices = PRICES.get(config.model, (0.0, 0.0))
        self.cost_usd = 0.0

    def add(self, completion: Completion) -> None:
        self.cost_usd += (
            completion.prompt_tokens * self._prices[0]
            + completion.completion_tokens * self._prices[1]
        )
