"""Market relevance filters to reduce non-financial noise before sentiment."""

from __future__ import annotations

import asyncio
import unicodedata
from dataclasses import dataclass
from typing import Protocol


class MarketRelevanceFilter(Protocol):
    def is_market_relevant(self, *, ticker: str, company_name: str, text: str) -> bool:
        ...


_KEYWORD_TERMS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "earnings",
    "revenue",
    "guidance",
    "shares",
    "share",
    "bullish",
    "bearish",
    "buy",
    "sell",
    "price target",
    "valuation",
    "p/e",
    "eps",
    "dividend",
    "analyst",
    "upgrade",
    "downgrade",
    "nasdaq",
    "nyse",
    "ipo",
    "quarterly",
    "q1",
    "q2",
    "q3",
    "q4",
}


class KeywordMarketRelevanceFilter:
    """Fast keyword/cashtag filter for market relevance."""

    def is_market_relevant(self, *, ticker: str, company_name: str, text: str) -> bool:
        normalized = text.lower()
        if f"${ticker.lower()}" in normalized:
            return True
        if ticker.lower() in normalized and any(term in normalized for term in _KEYWORD_TERMS):
            return True
        if company_name.lower() in normalized and any(term in normalized for term in _KEYWORD_TERMS):
            return True
        return False


@dataclass(frozen=True)
class HFModelRef:
    repo_id: str
    filename: str


@dataclass(frozen=True)
class RelevanceRequest:
    ticker: str
    company_name: str
    text: str


def parse_hf_gguf_ref(value: str) -> HFModelRef:
    """Parse `repo:filename_or_token` style arg.

    Kept for compatibility with existing workflows that pass
    `repo:IQ4_XS` style references to llama.cpp tools.
    """
    if ":" not in value:
        raise ValueError("Expected format '<repo_id>:<gguf-file-or-token>'")

    repo_id, suffix = value.split(":", 1)
    repo_id = repo_id.strip()
    suffix = suffix.strip()
    if not repo_id or not suffix:
        raise ValueError("Invalid value. Use '<repo_id>:<gguf-file-or-token>'")

    if suffix.endswith(".gguf"):
        return HFModelRef(repo_id=repo_id, filename=suffix)

    return HFModelRef(repo_id=repo_id, filename=f"*{suffix}*.gguf")


def parse_yes_no_response(text: str) -> bool:
    """Parse SI/NO style model output into market relevance boolean.
    
    Handles reasoning models that output <think>...</think> blocks.
    """
    # Remove reasoning block if present
    if "</think>" in text:
        text = text.split("</think>")[-1]
        
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()

    if "si" in normalized or "yes" in normalized:
        return True
    if "no" in normalized:
        return False

    print(f"Warning: Unexpected LLM relevance response '{text}'. Defaulting to relevant.")
    return True


async def classify_market_relevance_via_llama_server(
    requests: list[RelevanceRequest],
    *,
    api_url: str,
    concurrency: int = 10,
    timeout_seconds: float = 60.0,
) -> list[bool]:
    """Classify relevance using llama-server OpenAI-compatible API.

    Sends up to `concurrency` requests in parallel.
    """
    if not requests:
        return []

    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    results: list[bool | None] = [None] * len(requests)

    async def classify_one(index: int, item: RelevanceRequest, client: httpx.AsyncClient) -> None:
        prompt = (
            "Instrucción: Eres un clasificador binario. Responde EXACTAMENTE con 'SI' o 'NO' sin más texto.\n"
            "Determina si el texto habla del comportamiento bursatil, finanzas, inversion o mercado de la empresa.\n\n"
            f"Empresa: {item.company_name} ({item.ticker})\n"
            f"Texto: {item.text}\n\n"
            "Respuesta:"
        )

        payload = {
            "prompt": prompt,
            "n_predict": 256,
            "temperature": 0.0,
            "stop": []
        }

        async with semaphore:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            body = response.json()
            content = body.get("content", "")
            results[index] = parse_yes_no_response(content)

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*(classify_one(i, req, client) for i, req in enumerate(requests)))

    return [bool(value) for value in results]
