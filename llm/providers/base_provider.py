from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class LLMProvider(ABC):
    """Base interface for Large Language Model providers."""

    def __init__(self, name: str, models: Sequence[str]) -> None:
        self.name = name
        self.models = list(models)

    def supports(self, model: str) -> bool:
        return model in self.models

    def supports_images(self, model: str) -> bool:
        """Indicates whether this provider can generate images for the given model.

        The default implementation assumes no image support; providers that can
        generate images should override this.
        """
        return False

    @abstractmethod
    async def generate(self, model: str, messages: list[dict[str, str]], web_search: bool = False):
        """Return a text response for the provided conversation messages."""
