"""Process pending raw documents and save sentiment results.

Example:
    python3 -m pipeline.process_sentiment --limit 100
"""

from __future__ import annotations

import argparse

from nlp.cleaner import clean_text
from nlp.sentiment_analyzer import VaderSentimentAnalyzer
from storage.sqlite_store import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pending document sentiment")
    parser.add_argument("--db", default="data/sentiment.db")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    store = SQLiteStore(args.db)
    store.init_schema()
    analyzer = VaderSentimentAnalyzer()

    pending = store.list_documents_pending_sentiment(
        model_name=analyzer.model_name,
        limit=args.limit,
    )

    processed = 0
    for document in pending:
        result = analyzer.analyze(clean_text(document.text))
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
    print(f"Sentiment processing complete: model={analyzer.model_name}, processed={processed}")


if __name__ == "__main__":
    main()
