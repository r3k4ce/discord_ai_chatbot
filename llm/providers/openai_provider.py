from __future__ import annotations

from typing import Iterable, Union
import base64
import io
import aiohttp

from openai import AsyncOpenAI

from .base_provider import LLMProvider

_DEFAULT_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]


class OpenAIProvider(LLMProvider):
    """Adapter around OpenAI's chat completion API."""

    def __init__(
        self,
        api_key: str,
        models: Iterable[str] | None = None,
        client: AsyncOpenAI | None = None,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(name="openai", models=models or _DEFAULT_OPENAI_MODELS)
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._temperature = temperature

    async def generate(self, model: str, messages: list[dict[str, str]]) -> Union[str, dict]:
        if not self.supports(model):
            raise ValueError(f"Model '{model}' not supported by OpenAI provider")
        # If model is likely an image model, attempt the images endpoint first.
        if model == "gpt-image-1" or ("image" in model):
            # Last user prompt as image prompt
            prompt = None
            for m in reversed(messages):
                if m.get("role") == "user":
                    prompt = m.get("content")
                    break
            if prompt is None and messages:
                prompt = messages[-1].get("content")

            images_client = getattr(self._client, "images", None)
            if images_client is not None:
                response = None
                # Try shape generate(...)
                gen = getattr(images_client, "generate", None)
                try:
                    if callable(gen):
                        response = await gen(model=model, prompt=prompt)
                    elif gen and callable(getattr(gen, "create", None)):
                        response = await gen.create(model=model, prompt=prompt)
                except TypeError:
                    # If the signature is different, try common alt
                    pass

                # Also try images.create(...) fallback
                if response is None and callable(getattr(images_client, "create", None)):
                    try:
                        response = await images_client.create(model=model, prompt=prompt)
                    except Exception:
                        response = None

                # Parse response for base64 or url
                if response is not None:
                    data_list = getattr(response, "data", None) or (
                        response.get("data") if isinstance(response, dict) else None
                    )
                    if data_list:
                        first = data_list[0]
                        b64 = first.get("b64_json") if isinstance(first, dict) else getattr(first, "b64_json", None)
                        url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
                        if b64:
                            img_bytes = base64.b64decode(b64)
                            return {
                                "type": "image",
                                "data": img_bytes,
                                "filename": "image.png",
                                "content_type": "image/png",
                            }
                        if url:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(url) as resp:
                                    img_bytes = await resp.read()
                            return {
                                "type": "image",
                                "data": img_bytes,
                                "filename": "image.png",
                                "content_type": "image/png",
                            }

        # Default: use chat completions
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self._temperature,
        )
        message = response.choices[0].message.content
        return message.strip() if message else ""

    def supports_images(self, model: str) -> bool:
        # If the model name includes 'image' we consider it image-capable
        return "image" in model or model == "gpt-image-1"
