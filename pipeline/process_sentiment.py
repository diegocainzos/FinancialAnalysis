"""Process pending raw documents and save sentiment results.

Example:
    python3 -m pipeline.process_sentiment --limit 100
"""

from __future__ import annotations

import argparse
import asyncio

from nlp.cleaner import clean_text
from nlp.market_relevance import (
    KeywordMarketRelevanceFilter,
    RelevanceRequest,
    classify_market_relevance_via_llama_server,
)
from nlp.sentiment_analyzer import FinBertSentimentAnalyzer, SentimentAnalyzer, VaderSentimentAnalyzer
from storage.sqlite_store import SQLiteStore


def build_analyzer(*, model: str) -> SentimentAnalyzer:
    if model == "finbert":
        return FinBertSentimentAnalyzer()
    if model == "vader":
        return VaderSentimentAnalyzer()
    raise ValueError(f"Unsupported model: {model}")


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

    store = SQLiteStore(args.db)
    store.init_schema()
    analyzer = build_analyzer(model=args.model)

    companies_by_id = {c.id: c for c in store.list_companies() if c.id is not None}
    pending = store.list_documents_pending_sentiment(model_name=analyzer.model_name, limit=args.limit)

    processed = 0
    skipped_irrelevant = 0

    if args.relevance_filter == "llm":
        prepared: list[tuple[object, str]] = []
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

        relevance_flags = asyncio.run(
            classify_market_relevance_via_llama_server(
                requests,
                api_url=args.llama_api_url,
                concurrency=args.llama_concurrency,
            )
        )

        for (document, cleaned), is_relevant in zip(prepared, relevance_flags):
            if not is_relevant:
                skipped_irrelevant += 1
                continue

            result = analyzer.analyze(cleaned)
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
        keyword_filter = KeywordMarketRelevanceFilter() if args.relevance_filter == "keywords" else None

        for document in pending:
            company = companies_by_id.get(document.company_id)
            if company is None:
                continue

            cleaned = clean_text(document.text)
            if keyword_filter is not None:
                is_relevant = keyword_filter.is_market_relevant(
                    ticker=company.ticker,
                    company_name=company.name,
                    text=cleaned,
                )
                if not is_relevant:
                    skipped_irrelevant += 1
                    continue

            result = analyzer.analyze(cleaned)
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
    print(
        "Sentiment processing complete: "
        f"model={analyzer.model_name}, processed={processed}, skipped_irrelevant={skipped_irrelevant}"
    )


if __name__ == "__main__":
    main()
