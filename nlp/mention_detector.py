"""Rule-based company mention detection for the ingestion MVP."""

from __future__ import annotations

import re
from dataclasses import dataclass

from storage.sqlite_store import Company


@dataclass(frozen=True)
class Mention:
    company: Company
    matched_text: str
    match_type: str
    confidence: float


class MentionDetector:
    def detect(self, text: str, companies: list[Company]) -> list[Mention]:
        mentions: list[Mention] = []
        seen: set[tuple[int | None, str]] = set()

        for company in companies:
            candidates = [company.ticker, f"${company.ticker}", *company.aliases]
            for candidate in sorted(set(candidates), key=len, reverse=True):
                if not candidate:
                    continue
                if _contains(text, candidate):
                    key = (company.id, candidate.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    mentions.append(
                        Mention(
                            company=company,
                            matched_text=candidate,
                            match_type=_match_type(candidate, company),
                            confidence=_confidence(candidate, company),
                        )
                    )
        return mentions


def _contains(text: str, candidate: str) -> bool:
    escaped = re.escape(candidate)
    if candidate.startswith("$"):
        pattern = rf"(?<!\w){escaped}(?!\w)"
    elif candidate.isupper() and len(candidate) <= 5:
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = rf"\b{escaped}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _match_type(candidate: str, company: Company) -> str:
    if candidate == f"${company.ticker}":
        return "cashtag"
    if candidate.upper() == company.ticker.upper():
        return "ticker"
    if candidate.lower() == company.name.lower():
        return "company_name"
    return "alias"


def _confidence(candidate: str, company: Company) -> float:
    match_type = _match_type(candidate, company)
    return {
        "cashtag": 0.95,
        "ticker": 0.85,
        "company_name": 0.75,
        "alias": 0.6,
    }[match_type]
