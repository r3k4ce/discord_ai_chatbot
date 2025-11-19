from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

DEFAULT_MODELS: Dict[str, List[str]] = {
	"openai": ["gpt-5-mini", "gpt-5", "gpt-image-1"],
	"gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
}

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = DEFAULT_MODELS[DEFAULT_PROVIDER][0]


@dataclass(frozen=True)
class Settings:
	discord_token: str
	openai_api_key: str
	gemini_api_key: str
	database_path: str
	default_provider: str = DEFAULT_PROVIDER
	default_model: str = DEFAULT_MODEL
	model_presets: Dict[str, List[str]] = field(
		default_factory=lambda: deepcopy(DEFAULT_MODELS)
	)


def _env(key: str) -> str:
	value = os.getenv(key)
	if not value:
		raise RuntimeError(f"Environment variable {key} is required")
	return value


def load_settings() -> Settings:
	"""Load Settings from environment variables or .env file."""
	load_dotenv()

	discord_token = _env("DISCORD_BOT_TOKEN")
	openai_api_key = _env("OPENAI_API_KEY")
	gemini_api_key = _env("GEMINI_API_KEY")

	database_path = os.getenv("DATABASE_PATH", str(Path("chatbot.db").absolute()))
	default_provider = os.getenv("DEFAULT_PROVIDER", DEFAULT_PROVIDER).lower()
	models = deepcopy(DEFAULT_MODELS)
	default_model = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)

	provider_models = models.get(default_provider)
	if not provider_models:
		default_provider = DEFAULT_PROVIDER
		provider_models = models[DEFAULT_PROVIDER]
	if default_model not in provider_models:
		default_model = provider_models[0]

	return Settings(
		discord_token=discord_token,
		openai_api_key=openai_api_key,
		gemini_api_key=gemini_api_key,
		database_path=database_path,
		default_provider=default_provider,
		default_model=default_model,
		model_presets=models,
	)
