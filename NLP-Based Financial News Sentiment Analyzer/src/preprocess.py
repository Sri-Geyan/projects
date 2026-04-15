from __future__ import annotations
import re
import nltk

# Download once quietly; no-ops if already present
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords  # noqa: E402  (import after download)

_STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercase, strip non-alpha characters, remove English stopwords.
    Returns a space-joined token string ready for FinBERT input.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)