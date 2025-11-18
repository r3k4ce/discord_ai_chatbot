from __future__ import annotations

from typing import Dict

from utils.database_manager import DatabaseManager
from .providers.base_provider import LLMProvider


class LLMManager:
    """Routes prompts to the selected provider and keeps conversation context."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        providers: Dict[str, LLMProvider],
        default_provider: str,
        default_model: str,
    ) -> None:
        if default_provider not in providers:
            raise ValueError("Default provider must be in providers map")
        if not providers[default_provider].supports(default_model):
            raise ValueError("Default model must be available for the default provider")
        self._db_manager = db_manager
        self._providers = providers
        self._default_provider = default_provider
        self._default_model = default_model

    def get_available_models(self) -> dict[str, list[str]]:
        return {name: provider.models for name, provider in self._providers.items()}

    def _get_provider(self, name: str) -> LLMProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise ValueError(f"Unknown provider '{name}'")
        return provider

    async def set_user_model(self, user_id: int, provider_name: str, model_name: str) -> None:
        provider = self._get_provider(provider_name)
        if not provider.supports(model_name):
            raise ValueError(f"Invalid model '{model_name}' for provider '{provider_name}'")
        await self._db_manager.set_user_model(user_id, provider_name, model_name)

    async def get_user_model(self, user_id: int) -> tuple[str, str]:
        stored = await self._db_manager.get_user_model(user_id)
        if stored:
            provider_name, model_name = stored
            provider = self._providers.get(provider_name)
            if provider and provider.supports(model_name):
                return provider_name, model_name
        return self._default_provider, self._default_model

    async def generate_reply(self, user_id: int, prompt: str) -> str:
        provider_name, model_name = await self.get_user_model(user_id)
        provider = self._get_provider(provider_name)
        history = await self._db_manager.get_history(user_id)
        messages = history + [{"role": "user", "content": prompt}]
        await self._db_manager.append_history(user_id, "user", prompt)
        response = await provider.generate(model_name, messages)
        await self._db_manager.append_history(user_id, "assistant", response)
        return response

    async def generate_image(self, user_id: int, prompt: str) -> dict:
        provider_name, model_name = await self.get_user_model(user_id)
        provider = self._get_provider(provider_name)
        if not provider.supports_images(model_name):
            raise ValueError(
                f"Model '{model_name}' for provider '{provider_name}' does not support image generation"
            )
        # Do not include long chat history for images; use only the prompt
        messages = [{"role": "user", "content": prompt}]
        await self._db_manager.append_history(user_id, "user", prompt)
        response = await provider.generate(model_name, messages)
        # If an image was generated, append a small placeholder to history
        if isinstance(response, dict) and response.get("type") == "image":
            await self._db_manager.append_history(user_id, "assistant", "[image]")
            return response
        # Fall back: if provider returned text, record it
        if isinstance(response, str):
            await self._db_manager.append_history(user_id, "assistant", response)
            return {"type": "text", "text": response}
        raise RuntimeError("Unknown response from provider")
