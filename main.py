from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from config import Settings, load_settings
from llm.llm_manager import LLMManager
from llm.providers.base_provider import LLMProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.openai_provider import OpenAIProvider
from utils.database_manager import DatabaseManager
from cogs.chat import ChatCog
from cogs.settings import SettingsCog


class LLMCommandsBot(commands.Bot):
	def __init__(
		self,
		*,
		settings: Settings,
		db_manager: DatabaseManager,
		llm_manager: LLMManager,
	) -> None:
		intents = discord.Intents.default()
		intents.message_content = True
		super().__init__(command_prefix="!", intents=intents)
		self.settings = settings
		self.db_manager = db_manager
		self.llm_manager = llm_manager

	async def setup_hook(self) -> None:
		await self.add_cog(ChatCog(self, self.llm_manager))
		await self.add_cog(SettingsCog(self, self.llm_manager, self.db_manager))
		await self.tree.sync()


def _build_providers(settings: Settings) -> dict[str, LLMProvider]:
	providers: dict[str, LLMProvider] = {}
	openai_models = settings.model_presets.get("openai", [])
	gemini_models = settings.model_presets.get("gemini", [])
	providers["openai"] = OpenAIProvider(
		api_key=settings.openai_api_key,
		models=openai_models or None,
	)
	providers["gemini"] = GeminiProvider(
		api_key=settings.gemini_api_key,
		models=gemini_models or None,
	)
	return providers


def create_llm_manager(settings: Settings, db_manager: DatabaseManager) -> LLMManager:
	providers = _build_providers(settings)
	return LLMManager(
		db_manager=db_manager,
		providers=providers,
		default_provider=settings.default_provider,
		default_model=settings.default_model,
	)


def create_bot(
	settings: Settings,
	db_manager: DatabaseManager,
	llm_manager: LLMManager,
) -> commands.Bot:
	bot = LLMCommandsBot(settings=settings, db_manager=db_manager, llm_manager=llm_manager)

	@bot.event
	async def on_ready() -> None:  # pragma: no cover - network side effect
		logging.info("Bot connected as %s", bot.user)

	@bot.event
	async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
		if isinstance(error, commands.MissingRequiredArgument):
			await ctx.send("Missing arguments for this command. Please check !help.")
		else:
			await ctx.send(f"Unexpected error: {error}")

	return bot


async def run_bot() -> None:
	logging.basicConfig(level=logging.INFO)
	settings = load_settings()
	db_manager = DatabaseManager(settings.database_path)
	await db_manager.initialize()
	llm_manager = create_llm_manager(settings, db_manager)
	bot = create_bot(settings, db_manager, llm_manager)
	try:
		await bot.start(settings.discord_token)
	finally:
		await db_manager.close()


def main() -> None:
	asyncio.run(run_bot())


if __name__ == "__main__":
	main()
