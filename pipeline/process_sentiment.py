"""Process pending raw documents and save sentiment results.

Example:
    python3 -m pipeline.process_sentiment --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Awaitable, Callable

from nlp.cleaner import clean_text
from nlp.market_relevance import (
    KeywordMarketRelevanceFilter,
    RelevanceRequest,
    classify_market_relevance_via_llama_server,
)
from nlp.sentiment_analyzer import FinBertSentimentAnalyzer, SentimentAnalyzer, VaderSentimentAnalyzer
from storage.sqlite_store import PendingSentimentDocument, SQLiteStore


@dataclass(frozen=True)
class ProcessingStats:
    model_name: str
    pending: int
    llm_candidates: int
    processed: int
    skipped_irrelevant: int
    total_seconds: float
    keyword_filter_seconds: float
    llm_filter_seconds: float
    sentiment_seconds: float


def build_analyzer(*, model: str) -> SentimentAnalyzer:
    if model == "finbert":
        return FinBertSentimentAnalyzer()
    if model == "vader":
        return VaderSentimentAnalyzer()
    raise ValueError(f"Unsupported model: {model}")


RelevanceClassifier = Callable[..., Awaitable[list[bool]]]
LLAMA_RELEVANCE_CLASSIFIER = "llm:llama-server"


def process_pending_sentiment(
    *,
    db_path: str,
    limit: int,
    model: str,
    relevance_filter: str,
    llama_api_url: str,
    llama_concurrency: int,
    analyzer: SentimentAnalyzer | None = None,
    relevance_classifier: RelevanceClassifier = classify_market_relevance_via_llama_server,
) -> ProcessingStats:
    start = perf_counter()
    keyword_filter_seconds = 0.0
    llm_filter_seconds = 0.0
    sentiment_seconds = 0.0

    store = SQLiteStore(db_path)
    store.init_schema()
    analyzer = analyzer or build_analyzer(model=model)

    companies_by_id = {c.id: c for c in store.list_companies() if c.id is not None}
    exclude_irrelevant_classifier = LLAMA_RELEVANCE_CLASSIFIER if relevance_filter == "llm" else None
    pending = store.list_documents_pending_sentiment(
        model_name=analyzer.model_name,
        limit=limit,
        exclude_irrelevant_classifier=exclude_irrelevant_classifier,
    )

    processed = 0
    skipped_irrelevant = 0
    llm_candidates = 0

    if relevance_filter == "llm":
        prepared: list[tuple[PendingSentimentDocument, str]] = []
        requests: list[RelevanceRequest] = []

        for document in pending:
            company = companies_by_id.get(document.company_id)
            if company is None:
                continue

            cleaned = clean_text(document.text)
            prepared.append((document, cleaned))
            requests.append(
                RelevanceRequest(
                    ticker=company.ticker,
                    company_name=company.name,
                    text=cleaned,
                )
            )

        llm_candidates = len(requests)
        relevance_flags: list[bool] = []
        if requests:
            llm_start = perf_counter()
            relevance_flags = asyncio.run(
                relevance_classifier(
                    requests,
                    api_url=llama_api_url,
                    concurrency=llama_concurrency,
                )
            )
            llm_filter_seconds += perf_counter() - llm_start
            if len(relevance_flags) != len(requests):
                raise RuntimeError(
                    "LLM relevance classifier returned "
                    f"{len(relevance_flags)} results for {len(requests)} requests"
                )

        for (document, cleaned), is_relevant in zip(prepared, relevance_flags):
            store.save_document_relevance(
                company_id=document.company_id,
                raw_document_id=document.raw_document_id,
                classifier_name=LLAMA_RELEVANCE_CLASSIFIER,
                is_relevant=is_relevant,
            )
            if not is_relevant:
                skipped_irrelevant += 1
                continue

            sentiment_start = perf_counter()
            result = analyzer.analyze(cleaned)
            sentiment_seconds += perf_counter() - sentiment_start
            store.save_sentiment_result(
                company_id=document.company_id,
                raw_document_id=document.raw_document_id,
                sentiment_label=result.label,
                sentiment_score=result.score,
                confidence=result.confidence,
                model_name=result.model_name,
            )
            processed += 1

    else:
        keyword_filter = KeywordMarketRelevanceFilter() if relevance_filter == "keywords" else None

        for document in pending:
            company = companies_by_id.get(document.company_id)
            if company is None:
                continue

            cleaned = clean_text(document.text)
            if keyword_filter is not None:
                keyword_start = perf_counter()
                is_relevant = keyword_filter.is_market_relevant(
                    ticker=company.ticker,
                    company_name=company.name,
                    text=cleaned,
                )
                keyword_filter_seconds += perf_counter() - keyword_start
                if not is_relevant:
                    skipped_irrelevant += 1
                    continue

            sentiment_start = perf_counter()
            result = analyzer.analyze(cleaned)
            sentiment_seconds += perf_counter() - sentiment_start
            store.save_sentiment_result(
                company_id=document.company_id,
                raw_document_id=document.raw_document_id,
                sentiment_label=result.label,
                sentiment_score=result.score,
                confidence=result.confidence,
                model_name=result.model_name,
            )
            processed += 1

    store.close()
    return ProcessingStats(
        model_name=analyzer.model_name,
        pending=len(pending),
        llm_candidates=llm_candidates,
        processed=processed,
        skipped_irrelevant=skipped_irrelevant,
        total_seconds=perf_counter() - start,
        keyword_filter_seconds=keyword_filter_seconds,
        llm_filter_seconds=llm_filter_seconds,
        sentiment_seconds=sentiment_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pending document sentiment")
    parser.add_argument("--db", default="data/sentiment.db")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", choices=["finbert", "vader"], default="finbert")
    parser.add_argument(
        "--relevance-filter",
        choices=["none", "keywords", "llm"],
        default="keywords",
        help="Pre-filter documents that are not market-related before sentiment",
    )
    parser.add_argument(
        "--llama-api-url",
        default="http://127.0.0.1:8080/completion",
        help="llama-server native /completion endpoint",
    )
    parser.add_argument(
        "--llama-concurrency",
        type=int,
        default=1,
        help="Concurrent LLM relevance requests (match llama-server --parallel)",
    )
    args = parser.parse_args()

    stats = process_pending_sentiment(
        db_path=args.db,
        limit=args.limit,
        model=args.model,
        relevance_filter=args.relevance_filter,
        llama_api_url=args.llama_api_url,
        llama_concurrency=args.llama_concurrency,
    )

    print(
        "Sentiment processing complete: "
        f"model={stats.model_name}, pending={stats.pending}, "
        f"llm_candidates={stats.llm_candidates}, processed={stats.processed}, "
        f"skipped_irrelevant={stats.skipped_irrelevant}, "
        f"total_seconds={stats.total_seconds:.2f}, "
        f"keyword_seconds={stats.keyword_filter_seconds:.2f}, "
        f"llm_seconds={stats.llm_filter_seconds:.2f}, "
        f"sentiment_seconds={stats.sentiment_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
