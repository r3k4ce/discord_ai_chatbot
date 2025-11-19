from __future__ import annotations

from discord import app_commands
from discord.ext import commands
import discord

from llm.llm_manager import LLMManager
from utils.database_manager import DatabaseManager


class SettingsCog(commands.Cog, name="Settings"):
    """Commands for adjusting model preferences and history."""

    def __init__(
        self,
        bot: commands.Bot,
        llm_manager: LLMManager,
        db_manager: DatabaseManager,
    ) -> None:
        self.bot = bot
        self.llm_manager = llm_manager
        self.db_manager = db_manager

    @app_commands.command(name="setmodel", description="Set the provider and model to use")
    @app_commands.describe(provider="The LLM provider (e.g., openai, gemini)", model="The model name")
    async def setmodel(self, interaction: discord.Interaction, provider: str, model: str) -> None:
        try:
            await self.llm_manager.set_user_model(interaction.user.id, provider.lower(), model)
        except ValueError as exc:
            await interaction.response.send_message(content=f"Invalid model selection: {exc}", ephemeral=True)
            return
        await interaction.response.send_message(content=f"✅ Model set to {provider.lower()}:{model}")

    @app_commands.command(name="clearchat", description="Clear your saved conversation history")
    async def clearchat(self, interaction: discord.Interaction) -> None:
        await self.db_manager.clear_history(interaction.user.id)
        await interaction.response.send_message(content="🧹 Cleared your conversation history.", ephemeral=True)

    @app_commands.command(name="models", description="List the available providers and models")
    async def models(self, interaction: discord.Interaction) -> None:
        models = self.llm_manager.get_available_models()
        lines = [
            f"**{provider}**: {', '.join(model_list)}" for provider, model_list in models.items()
        ]
        await interaction.response.send_message(content="Available models:\n" + "\n".join(lines), ephemeral=True)
