from __future__ import annotations

from discord.ext import commands

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

    @commands.command(name="setmodel", help="Set the provider and model to use")
    async def setmodel(self, ctx: commands.Context, provider: str, model: str) -> None:
        try:
            await self.llm_manager.set_user_model(ctx.author.id, provider.lower(), model)
        except ValueError as exc:
            await ctx.send(content=f"Invalid model selection: {exc}")
            return
        await ctx.send(content=f"✅ Model set to {provider.lower()}:{model}")

    @commands.command(name="clearchat", help="Clear your saved conversation history")
    async def clearchat(self, ctx: commands.Context) -> None:
        await self.db_manager.clear_history(ctx.author.id)
        await ctx.send(content="🧹 Cleared your conversation history.")

    @commands.command(name="models", help="List the available providers and models")
    async def models(self, ctx: commands.Context) -> None:
        models = self.llm_manager.get_available_models()
        lines = [
            f"**{provider}**: {', '.join(model_list)}" for provider, model_list in models.items()
        ]
        await ctx.send(content="Available models:\n" + "\n".join(lines))
