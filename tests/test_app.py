import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import base64
import io

import discord
import pytest
import pytest_asyncio
from discord.ext import commands

from config import Settings, load_settings
from utils.database_manager import DatabaseManager
from llm.llm_manager import LLMManager
from llm.providers.base_provider import LLMProvider
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.gemini_provider import GeminiProvider
from cogs.chat import ChatCog
from utils.text_utils import chunk_text, DISCORD_MAX_MESSAGE_LENGTH
from cogs.settings import SettingsCog
import main


class DummyProvider(LLMProvider):
    """Simple LLM provider used for testing manager logic."""

    def __init__(self):
        super().__init__(name="dummy", models=["dummy-model"])
        self.calls = []

    async def generate(self, model: str, messages: list[dict], web_search: bool = False) -> str:
        self.calls.append((model, messages))
        return "dummy-response"


class DummyImageProvider(LLMProvider):
    def __init__(self):
        super().__init__(name="dummy-image", models=["dummy-image-model"])

    async def generate(self, model: str, messages: list[dict], web_search: bool = False) -> dict:
        # Return an image dict for testing
        return {"type": "image", "data": b"PNG", "filename": "image.png"}

    def supports_images(self, model: str) -> bool:
        return model in self.models


class SpyProvider(LLMProvider):
    def __init__(self):
        super().__init__(name="spy", models=["spy-model"]) 
        self.calls = []

    async def generate(self, model: str, messages: list[dict], web_search: bool = False):
        self.calls.append((model, messages, web_search))
        return "spy-response"


@pytest.fixture
def temp_env(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-discord")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "chatbot.db"))
    monkeypatch.setenv("DEFAULT_PROVIDER", "dummy")
    monkeypatch.setenv("DEFAULT_MODEL", "dummy-model")
    return tmp_path


def test_load_settings_reads_environment(temp_env):
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.discord_token == "test-discord"
    assert settings.openai_api_key == "test-openai"
    assert settings.gemini_api_key == "test-gemini"
    assert Path(settings.database_path).name == "chatbot.db"
    assert "openai" in settings.model_presets
    assert "gemini" in settings.model_presets


@pytest_asyncio.fixture
async def db_manager(tmp_path):
    manager = DatabaseManager(tmp_path / "app.db")
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.mark.asyncio
async def test_database_user_model_roundtrip(db_manager):
    await db_manager.set_user_model(1, "dummy", "dummy-model")
    provider, model = await db_manager.get_user_model(1)
    assert provider == "dummy"
    assert model == "dummy-model"


@pytest.mark.asyncio
async def test_database_history_flow(db_manager):
    await db_manager.append_history(2, "user", "Hello")
    await db_manager.append_history(2, "assistant", "Hi there")
    history = await db_manager.get_history(2)
    assert len(history) == 2
    assert history[0]["content"] == "Hello"
    await db_manager.clear_history(2)
    assert await db_manager.get_history(2) == []


@pytest_asyncio.fixture
async def dummy_llm_manager(db_manager):
    providers = {"dummy": DummyProvider()}
    manager = LLMManager(
        db_manager=db_manager,
        providers=providers,
        default_provider="dummy",
        default_model="dummy-model",
    )
    return manager


@pytest_asyncio.fixture
async def spy_manager(db_manager):
    providers = {"spy": SpyProvider()}
    manager = LLMManager(db_manager=db_manager, providers=providers, default_provider="spy", default_model="spy-model")
    return manager


@pytest_asyncio.fixture
async def dummy_image_manager(db_manager):
    providers = {"dummy-image": DummyImageProvider()}
    manager = LLMManager(
        db_manager=db_manager,
        providers=providers,
        default_provider="dummy-image",
        default_model="dummy-image-model",
    )
    return manager


@pytest.mark.asyncio
async def test_llm_manager_generate_reply_records_history(dummy_llm_manager, db_manager):
    response = await dummy_llm_manager.generate_reply(10, "Ping?")
    assert response == "dummy-response"
    stored = await db_manager.get_history(10)
    assert stored[-1]["role"] == "assistant"
    assert stored[-1]["content"] == "dummy-response"


@pytest.mark.asyncio
async def test_llm_manager_generate_image_records_history(dummy_image_manager, db_manager):
    response = await dummy_image_manager.generate_image(20, "Make a scene")
    assert isinstance(response, dict)
    assert response["type"] == "image"
    # Verify history placeholder for assistant
    history = await db_manager.get_history(20)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "[image]"


@pytest.mark.asyncio
async def test_llm_manager_validates_models(dummy_llm_manager):
    with pytest.raises(ValueError):
        await dummy_llm_manager.set_user_model(11, "dummy", "missing-model")


@pytest.mark.asyncio
async def test_llm_manager_available_models(dummy_llm_manager):
    @pytest.mark.asyncio
    async def test_llm_manager_passes_web_search_flag(spy_manager, db_manager):
        await db_manager.set_user_model(1, "spy", "spy-model")
        await db_manager.set_user_web_search(1, True)
        res = await spy_manager.generate_reply(1, "Ping with web search")
        assert res == "spy-response"
        assert spy_manager._providers["spy"].calls[0][2] is True
    models = dummy_llm_manager.get_available_models()
    assert models == {"dummy": ["dummy-model"]}


@pytest.mark.asyncio
async def test_openai_provider_invokes_client():
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "openai says hi"
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider(api_key="test", client=mock_client, models=["gpt-4o-mini", "gpt-image-1"])
    result = await provider.generate("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert result == "openai says hi"
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_provider_image_response():
    # Mock an images.generate returning base64 b64_json
    png_bytes = b"PNGDATA"
    b64 = base64.b64encode(png_bytes).decode()
    mock_response = MagicMock()
    mock_response.data = [{"b64_json": b64}]

    mock_images = MagicMock()
    mock_images.generate = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.images = mock_images

    provider = OpenAIProvider(api_key="test", client=mock_client, models=["gpt-image-1"]) 
    result = await provider.generate("gpt-image-1", [{"role": "user", "content": "draw X"}])
    assert isinstance(result, dict)
    assert result["type"] == "image"
    assert result["data"] == png_bytes


class DummyGeminiModel:
    def __init__(self, text: str):
        self._response = MagicMock()
        self._response.text = text
        self.generate_content = MagicMock(return_value=self._response)


async def direct_runner(fn, prompt):
    return fn(prompt)


@pytest.mark.asyncio
async def test_gemini_provider_invokes_model():
    model = DummyGeminiModel("gemini says hi")

    def factory(model_name: str):
        assert model_name == "gemini-pro"
        return model

    provider = GeminiProvider(api_key="test", model_factory=factory, runner=direct_runner)
    result = await provider.generate("gemini-pro", [{"role": "user", "content": "hi"}])
    assert result == "gemini says hi"
    model.generate_content.assert_called_once()


@pytest_asyncio.fixture
async def bot():
    intents = discord.Intents.none()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    yield bot
    await bot.close()


def mock_interaction(user_id=42):
    interaction = MagicMock()
    interaction.user.id = user_id
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


@pytest.mark.asyncio
async def test_chat_command_sends_reply(bot, dummy_llm_manager):
    cog = ChatCog(bot, dummy_llm_manager)
    interaction = mock_interaction(42)
    await ChatCog.chat.callback(cog, interaction, message="Hello")
    interaction.response.defer.assert_awaited_once()
    interaction.followup.send.assert_awaited_once()
    assert interaction.followup.send.await_args.kwargs["content"] == "dummy-response"


@pytest.mark.asyncio
async def test_chat_command_chunks_long_reply(bot, dummy_llm_manager):
    # Create a long response exceeding Discord's 2000-char limit
    long_text = "A" * (DISCORD_MAX_MESSAGE_LENGTH * 2 + 300)
    cog = ChatCog(bot, dummy_llm_manager)
    interaction = mock_interaction(123)
    # Override the manager to return the long text
    cog.llm_manager.generate_reply = AsyncMock(return_value=long_text)
    await ChatCog.chat.callback(cog, interaction, message="Hello")

    # Ensure multiple sends occurred and the concatenated content equals the long text
    expected_chunks = list(chunk_text(long_text))
    assert interaction.followup.send.await_count == len(expected_chunks)
    sent = "".join(call.kwargs["content"] for call in interaction.followup.send.await_args_list)
    assert sent == long_text


@pytest.mark.asyncio
async def test_chat_command_sends_image(bot, dummy_llm_manager):
    cog = ChatCog(bot, dummy_llm_manager)
    interaction = mock_interaction(42)
    data = b"IMG"
    cog.llm_manager.generate_reply = AsyncMock(return_value={"type": "image", "data": data, "filename": "image.png"})
    await ChatCog.chat.callback(cog, interaction, message="Generate an image")
    interaction.followup.send.assert_awaited_once()
    assert "file" in interaction.followup.send.await_args.kwargs
    sent_file = interaction.followup.send.await_args.kwargs["file"]
    assert getattr(sent_file, "filename", None) == "image.png"


@pytest.mark.asyncio
async def test_image_command_sends_file(bot, dummy_image_manager):
    cog = ChatCog(bot, dummy_image_manager)
    interaction = mock_interaction(84)
    await ChatCog.image.callback(cog, interaction, prompt="Create sunset")
    interaction.followup.send.assert_awaited_once()
    sent = interaction.followup.send.await_args.kwargs
    assert "file" in sent
    file_obj = sent["file"]
    assert getattr(file_obj, "filename", None) == "image.png"


@pytest.mark.asyncio
async def test_chat_command_reports_errors(bot, dummy_llm_manager):
    cog = ChatCog(bot, dummy_llm_manager)
    interaction = mock_interaction(42)
    cog.llm_manager.generate_reply = AsyncMock(side_effect=RuntimeError("boom"))
    await ChatCog.chat.callback(cog, interaction, message="Hello")
    interaction.followup.send.assert_awaited_once()
    assert "boom" in interaction.followup.send.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_set_model_command_success(bot, dummy_llm_manager, db_manager):
    cog = SettingsCog(bot, dummy_llm_manager, db_manager)
    interaction = mock_interaction(7)
    await SettingsCog.setmodel.callback(cog, interaction, provider="dummy", model="dummy-model")
    interaction.response.send_message.assert_awaited_once()
    assert "dummy" in interaction.response.send_message.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_set_model_command_validation(bot, dummy_llm_manager, db_manager):
    cog = SettingsCog(bot, dummy_llm_manager, db_manager)
    interaction = mock_interaction(7)
    await SettingsCog.setmodel.callback(cog, interaction, provider="dummy", model="wrong")
    interaction.response.send_message.assert_awaited_once()
    assert "Invalid" in interaction.response.send_message.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_set_websearch_command_success(bot, dummy_llm_manager, db_manager):
    cog = SettingsCog(bot, dummy_llm_manager, db_manager)
    interaction = mock_interaction(7)
    await SettingsCog.websearch.callback(cog, interaction, enabled=True)
    interaction.response.send_message.assert_awaited_once()
    assert await db_manager.get_user_web_search(7) is True


@pytest.mark.asyncio
async def test_clear_chat_command(bot, dummy_llm_manager, db_manager):
    cog = SettingsCog(bot, dummy_llm_manager, db_manager)
    interaction = mock_interaction(99)
    await db_manager.append_history(99, "user", "Hello")
    await SettingsCog.clearchat.callback(cog, interaction)
    interaction.response.send_message.assert_awaited_once()
    assert await db_manager.get_history(99) == []


@pytest.mark.asyncio
async def test_list_models_command(bot, dummy_llm_manager, db_manager):
    cog = SettingsCog(bot, dummy_llm_manager, db_manager)
    interaction = mock_interaction(99)
    await SettingsCog.models.callback(cog, interaction)
    interaction.response.send_message.assert_awaited_once()
    assert "dummy-model" in interaction.response.send_message.await_args.kwargs["content"]


@pytest_asyncio.fixture
async def settings(tmp_path):
    return Settings(
        discord_token="token",
        openai_api_key="openai",
        gemini_api_key="gemini",
        database_path=str(tmp_path / "bot.db"),
        default_provider="dummy",
        default_model="dummy-model",
        model_presets={"dummy": ["dummy-model"]},
    )


@pytest.mark.asyncio
async def test_create_bot_registers_cogs(settings, db_manager, dummy_llm_manager):
    bot = main.create_bot(settings, db_manager, dummy_llm_manager)
    bot.tree.sync = AsyncMock()
    try:
        await bot.setup_hook()
        assert "Chat" in bot.cogs
        assert "Settings" in bot.cogs
        bot.tree.sync.assert_awaited_once()
    finally:
        await bot.close()
