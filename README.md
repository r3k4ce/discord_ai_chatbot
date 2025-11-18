# Discord LLM Chatbot

A Discord bot that lets users hold contextual conversations with multiple LLM
providers (OpenAI + Gemini). It persists user preferences and conversation
history in SQLite so you can pick up where you left off, and it exposes
straightforward commands for quick experimentation.

## Features
- `!chat <message>` streams your prompt and rich history into the currently
	selected model and replies inline.
 - `!image <prompt>` generates an image using your selected provider/model (if supported) and replies with an attached file.
- `!setmodel <provider> <model>` switches between hard-coded, vetted model
	options per provider.
- `!models` lists the available providers/models so users know what is enabled.
- `!clearchat` wipes the stored history for the requesting user.
- Provider layer cleanly abstracts OpenAI (via `AsyncOpenAI`) and Gemini (via
	`google-generativeai`), making it easy to add more vendors later.
 - Image-capable models (like `gpt-image-1`) are supported; set them with `!setmodel openai gpt-image-1` and use `!image` to generate images. The image is sent as an attachment in the chat. Note: image-capable models may only work with specific provider endpoints.
- End-to-end pytest suite (`tests/` plus the `tests.py` runner) exercises every
	module before you ever connect to Discord.

## Prerequisites
- Python 3.12+
- Discord bot token with the Message Content intent enabled
- OpenAI API key and Gemini API key

## Initial Setup
1. Create the virtual environment if it does not exist yet:
	 ```bash
	 python -m venv venv
	 ```
2. Install dependencies inside the virtualenv:
3. Copy `.env.example` to `.env` and provide the required secrets:
	 ```bash
	 cp .env.example .env
	 ```
	 Required variables:
	 - `DISCORD_BOT_TOKEN`
	 - `OPENAI_API_KEY`
	 - `GEMINI_API_KEY`
	 Optional overrides: `DATABASE_PATH`, `COMMAND_PREFIX`, `DEFAULT_PROVIDER`,
	 `DEFAULT_MODEL`.

## Running the Bot
```bash
source venv/bin/activate
python main.py
```
The bot registers the `!chat`, `!setmodel`, `!models`, and `!clearchat`
commands. Keep the process alive on your personal server to stay connected.

## Testing
Run the complete suite (includes database, provider, cog, and wiring tests):
```bash
pytest
```
