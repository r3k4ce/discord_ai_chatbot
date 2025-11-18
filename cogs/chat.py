from __future__ import annotations

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

    @commands.command(name="chat", help="Send a prompt to the configured model")
    async def chat(self, ctx: commands.Context, *, message: str) -> None:
        if not message:
            await ctx.send(content="Please include a message after !chat.")
            return
        try:
            async with ctx.typing():
                reply = await self.llm_manager.generate_reply(ctx.author.id, message)
        except Exception as exc:  # pragma: no cover - exercised via tests
            await ctx.send(content=f"⚠️ Failed to contact the LLM: {exc}")
            return
        # If the provider returned an image dict, send as a Discord file
        if isinstance(reply, dict) and reply.get("type") == "image":
            data = reply.get("data")
            filename = reply.get("filename", "image.png")
            await ctx.send(file=discord.File(io.BytesIO(data), filename))
            return

        # Stream the reply in chunks so we never exceed Discord's 2000 char
        # message length limit. We don't label chunks; they are emitted in
        # sequence to preserve conversational flow.
        for chunk in chunk_text(reply):
            await ctx.send(content=chunk)

    @commands.command(name="image", help="Generate an image with the configured model")
    async def image(self, ctx: commands.Context, *, prompt: str) -> None:
        if not prompt:
            await ctx.send(content="Please include a prompt after !image.")
            return
        try:
            async with ctx.typing():
                image_response = await self.llm_manager.generate_image(ctx.author.id, prompt)
        except Exception as exc:  # pragma: no cover - exercised via tests
            await ctx.send(content=f"⚠️ Failed to contact the LLM: {exc}")
            return
        if isinstance(image_response, dict) and image_response.get("type") == "image":
            data = image_response.get("data")
            filename = image_response.get("filename", "image.png")
            await ctx.send(content="✅ Generated image:", file=discord.File(io.BytesIO(data), filename))
            return
        # Fallback in case provider returned text
        if isinstance(image_response, dict) and image_response.get("type") == "text":
            for chunk in chunk_text(image_response.get("text", "")):
                await ctx.send(content=chunk)
