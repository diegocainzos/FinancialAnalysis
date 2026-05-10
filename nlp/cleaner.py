"""Text normalization for sentiment analysis."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean provider text while preserving cashtags and tickers."""
    text = _URL_RE.sub(" ", text)
    text = text.replace("\u200b", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
