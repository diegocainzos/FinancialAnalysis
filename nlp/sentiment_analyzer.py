"""Sentiment analyzer interface and implementations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    confidence: float
    model_name: str


class SentimentAnalyzer(Protocol):
    model_name: str

    def analyze(self, text: str) -> SentimentResult:
        ...


class FinBertSentimentAnalyzer:
    """Hugging Face FinBERT sentiment analyzer.

    Model: ProsusAI/finbert
    """

    def __init__(
        self,
        *,
        model_id: str = "ProsusAI/finbert",
        classifier: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = f"finbert:{model_id}"

        if classifier is not None:
            self._classifier = classifier
            return

        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "transformers is required to use FinBERT sentiment analyzer"
            ) from exc

        try:
            self._classifier = pipeline(
                task="text-classification",
                model=model_id,
                tokenizer=model_id,
            )
        except OSError as exc:  # pragma: no cover - model download/load error path
            raise RuntimeError(
                f"Unable to load FinBERT model '{model_id}'. "
                "Verify internet access or local cache."
            ) from exc

    def analyze(self, text: str) -> SentimentResult:
        raw = self._classifier(text, truncation=True, max_length=512)
        prediction = raw[0] if isinstance(raw, list) else raw

        label = _normalize_finbert_label(str(prediction["label"]))
        confidence = float(prediction.get("score", 0.0))
        score = _finbert_score(label, confidence)

        return SentimentResult(
            label=label,
            score=score,
            confidence=confidence,
            model_name=self.model_name,
        )


class VaderSentimentAnalyzer:
    """VADER-style analyzer with dependency-free fallback."""

    model_name = "vader-fallback"

    def __init__(self) -> None:
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            return
        self._vader = SentimentIntensityAnalyzer()
        self.model_name = "vader"

    def analyze(self, text: str) -> SentimentResult:
        if self._vader is not None:
            compound = float(self._vader.polarity_scores(text)["compound"])
        else:
            compound = _fallback_compound(text)

        label = _label_from_score(compound)
        return SentimentResult(
            label=label,
            score=compound,
            confidence=_confidence_from_score(compound, label),
            model_name=self.model_name,
        )


def _normalize_finbert_label(label: str) -> str:
    normalized = label.strip().lower()
    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported FinBERT label: {label}")
    return mapping[normalized]


def _finbert_score(label: str, confidence: float) -> float:
    if label == "positive":
        return min(1.0, max(0.0, confidence))
    if label == "negative":
        return -min(1.0, max(0.0, confidence))
    return 0.0


def _label_from_score(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _confidence_from_score(score: float, label: str) -> float:
    if label == "neutral":
        return max(0.5, 1.0 - min(abs(score), 1.0))
    return min(1.0, max(0.5, abs(score)))


_POSITIVE_TERMS = {
    "up",
    "gain",
    "gains",
    "growth",
    "beat",
    "beats",
    "bullish",
    "buy",
    "strong",
    "surge",
    "rally",
    "profit",
    "profits",
    "record",
    "outperform",
    "upgrade",
    "positive",
    "moon",
}

_NEGATIVE_TERMS = {
    "down",
    "drop",
    "drops",
    "fall",
    "falls",
    "loss",
    "losses",
    "miss",
    "misses",
    "bearish",
    "sell",
    "weak",
    "crash",
    "plunge",
    "downgrade",
    "negative",
    "lawsuit",
    "risk",
    "risks",
}

_NEGATIONS = {"not", "no", "never", "without"}


def _fallback_compound(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0

    score = 0
    hits = 0
    for index, token in enumerate(tokens):
        polarity = 0
        if token in _POSITIVE_TERMS:
            polarity = 1
        elif token in _NEGATIVE_TERMS:
            polarity = -1
        if polarity == 0:
            continue

        if index > 0 and tokens[index - 1] in _NEGATIONS:
            polarity *= -1
        score += polarity
        hits += 1

    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(3, hits)))
