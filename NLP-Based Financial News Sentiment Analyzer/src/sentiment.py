from __future__ import annotations
from functools import lru_cache
from transformers import pipeline as hf_pipeline


@lru_cache(maxsize=1)
def _get_pipeline():
    """Lazy-load FinBERT once and cache it. Avoids cold-start crashes on import."""
    return hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
    )


def get_sentiment(text: str) -> float:
    """
    Score a financial headline with FinBERT.

    Returns a signed float in [-1.0, +1.0]:
      +confidence  for positive
       0.0         for neutral
      -confidence  for negative
    """
    if not text or not text.strip():
        return 0.0

    result = _get_pipeline()(text[:512])[0]   # truncate to BERT max
    label = result["label"].lower()
    score = float(result["score"])

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    return 0.0