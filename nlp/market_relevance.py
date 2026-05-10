"""Market relevance filters to reduce non-financial noise before sentiment."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


class MarketRelevanceFilter(Protocol):
    def is_market_relevant(self, *, ticker: str, company_name: str, text: str) -> bool:
        ...


_MARKET_TERMS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "earnings",
    "revenue",
    "sales estimates",
    "shares",
    "share price",
    "bullish",
    "bearish",
    "buy rating",
    "sell rating",
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
    "fiscal",
    "q1",
    "q2",
    "q3",
    "q4",
}

_WEAK_MARKET_TERMS = {
    "guidance",
    "buy",
    "sell",
    "share",
}

_CONFIRMING_MARKET_TERMS = {
    "stock",
    "stocks",
    "earnings",
    "revenue",
    "eps",
    "quarter",
    "quarterly",
    "fiscal",
    "price target",
    "valuation",
    "analyst",
    "nasdaq",
    "nyse",
}


class KeywordMarketRelevanceFilter:
    """Fast keyword/cashtag filter for market relevance."""

    def is_market_query(self, *, ticker: str, company_name: str, query: str) -> bool:
        normalized = query.lower()
        if f"${ticker.lower()}" in normalized:
            return True
        if ticker.lower() in normalized and any(term in normalized for term in _MARKET_TERMS):
            return True
        if company_name.lower() in normalized and any(term in normalized for term in _MARKET_TERMS):
            return True
        return False

    def is_market_relevant(self, *, ticker: str, company_name: str, text: str) -> bool:
        normalized = text.lower()
        if f"${ticker.lower()}" in normalized:
            return True
        if ticker.lower() in normalized and any(term in normalized for term in _MARKET_TERMS):
            return True
        if company_name.lower() in normalized and any(term in normalized for term in _MARKET_TERMS):
            return True
        if (
            (ticker.lower() in normalized or company_name.lower() in normalized)
            and any(term in normalized for term in _WEAK_MARKET_TERMS)
            and any(term in normalized for term in _CONFIRMING_MARKET_TERMS)
        ):
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
    tokens = re.findall(r"[a-z]+", normalized)

    if any(token in {"si", "yes"} for token in tokens):
        return True
    if "no" in tokens:
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
            "Clasifica si el texto habla de bolsa, resultados trimestrales corporativos, ingresos, "
            "acciones, valoración o mercado de la empresa.\n"
            "NO cuentes usos de producto, soporte, música, coches, empleo o ingresos personales.\n"
            "Responde solo SI o NO.\n\n"
            "Empresa: Tesla (TSLA)\n"
            "Texto: I love driving my Tesla car.\n"
            "Respuesta: NO\n\n"
            "Empresa: Meta (META)\n"
            "Texto: Meta promotes a side hustle promising monthly earnings for creators.\n"
            "Respuesta: NO\n\n"
            "Empresa: Nvidia (NVDA)\n"
            "Texto: NVDA shares fall after quarterly earnings.\n"
            "Respuesta: SI\n\n"
            f"Empresa: {item.company_name} ({item.ticker})\n"
            f"Texto: {item.text}\n"
            "Respuesta:"
        )

        payload = {
            "prompt": prompt,
            "n_predict": 4,
            "temperature": 0.0,
            "stop": ["\n"]
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
