from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable

import google.generativeai as genai

from .base_provider import LLMProvider

_DEFAULT_GEMINI_MODELS = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

Runner = Callable[[Callable[..., object], str], Awaitable[object]]


def _default_runner(fn: Callable[..., object], prompt: str) -> Awaitable[object]:
    return asyncio.to_thread(fn, prompt)


class GeminiProvider(LLMProvider):
    """Adapter around the Google Gemini SDK."""

    def __init__(
        self,
        api_key: str,
        models: Iterable[str] | None = None,
        model_factory: Callable[[str], object] | None = None,
        runner: Runner | None = None,
    ) -> None:
        super().__init__(name="gemini", models=models or _DEFAULT_GEMINI_MODELS)
        if model_factory is None:
            genai.configure(api_key=api_key)
            self._model_factory = lambda model_name: genai.GenerativeModel(model_name)
        else:
            self._model_factory = model_factory
        self._runner = runner or _default_runner

    async def generate(self, model: str, messages: list[dict[str, str]]) -> str:
        if not self.supports(model):
            raise ValueError(f"Model '{model}' not supported by Gemini provider")
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        model_instance = self._model_factory(model)
        response = await self._runner(model_instance.generate_content, prompt)
        text = getattr(response, "text", "")
        return text.strip()
