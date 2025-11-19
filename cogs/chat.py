from __future__ import annotations

from discord import app_commands
from discord.ext import commands
import discord
import io

from utils.text_utils import chunk_text

from llm.llm_manager import LLMManager


class ChatCog(commands.Cog, name="Chat"):
    """Primary chat entrypoint for users."""

    def __init__(self, bot: commands.Bot, llm_manager: LLMManager) -> None:
        self.bot = bot
        self.llm_manager = llm_manager

    @app_commands.command(name="chat", description="Send a prompt to the configured model")
    @app_commands.describe(message="The message to send to the LLM")
    async def chat(self, interaction: discord.Interaction, message: str) -> None:
        await interaction.response.defer(thinking=True)
        try:
            reply = await self.llm_manager.generate_reply(interaction.user.id, message)
        except Exception as exc:  # pragma: no cover - exercised via tests
            await interaction.followup.send(content=f"⚠️ Failed to contact the LLM: {exc}")
            return
        # If the provider returned an image dict, send as a Discord file
        if isinstance(reply, dict) and reply.get("type") == "image":
            data = reply.get("data")
            filename = reply.get("filename", "image.png")
            await interaction.followup.send(file=discord.File(io.BytesIO(data), filename))
            return

        # Stream the reply in chunks so we never exceed Discord's 2000 char
        # message length limit. We don't label chunks; they are emitted in
        # sequence to preserve conversational flow.
        for chunk in chunk_text(reply):
            await interaction.followup.send(content=chunk)

    @app_commands.command(name="image", description="Generate an image with the configured model")
    @app_commands.describe(prompt="The prompt for the image generation")
    async def image(self, interaction: discord.Interaction, prompt: str) -> None:
        await interaction.response.defer(thinking=True)
        try:
            image_response = await self.llm_manager.generate_image(interaction.user.id, prompt)
        except Exception as exc:  # pragma: no cover - exercised via tests
            await interaction.followup.send(content=f"⚠️ Failed to contact the LLM: {exc}")
            return
        if isinstance(image_response, dict) and image_response.get("type") == "image":
            data = image_response.get("data")
            filename = image_response.get("filename", "image.png")
            await interaction.followup.send(content="✅ Generated image:", file=discord.File(io.BytesIO(data), filename))
            return
        # Fallback in case provider returned text
        if isinstance(image_response, dict) and image_response.get("type") == "text":
            for chunk in chunk_text(image_response.get("text", "")):
                await interaction.followup.send(content=chunk)
