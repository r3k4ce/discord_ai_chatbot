from __future__ import annotations

from typing import Generator


DISCORD_MAX_MESSAGE_LENGTH = 2000


def chunk_text(text: str, limit: int = DISCORD_MAX_MESSAGE_LENGTH) -> Generator[str, None, None]:
    """Yield chunks of at most `limit` characters from `text`.

    This preserves order and attempts to break on newlines if possible, but
    will fall back to fixed-size chunks when necessary.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    start = 0
    n = len(text)
    while start < n:
        if n - start <= limit:
            yield text[start:n]
            break
        # Try to split at last newline within limit to avoid cutting sentences.
        end = start + limit
        if "\n" in text[start:end]:
            last_nl = text.rfind("\n", start, end)
            if last_nl != -1 and last_nl > start:
                yield text[start:last_nl]
                start = last_nl + 1
                continue
        # Otherwise, split at limit
        yield text[start:end]
        start = end
