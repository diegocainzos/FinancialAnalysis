"""CLI entrypoint for Bluesky company-post ingestion.

Example:
    python3 -m pipeline.ingest_bluesky --limit 20
    python3 -m pipeline.ingest_bluesky --company TSLA --limit 50 --lang en
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from ingestion.bluesky_client import BlueskyClient, BlueskyPost
from nlp.mention_detector import MentionDetector
from storage.sqlite_store import SQLiteStore, load_companies_from_json


@dataclass(frozen=True)
class SearchResult:
    company_id: int
    query: str
    posts: list[BlueskyPost]


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Ingest company mentions from Bluesky")
    parser.add_argument("--companies", default="config/companies.json")
    parser.add_argument("--db", default="data/sentiment.db")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--company", help="Optional ticker filter, e.g. TSLA")
    parser.add_argument("--lang", help="Optional Bluesky language filter, e.g. en or es")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    store = SQLiteStore(args.db)
    store.init_schema()
    store.upsert_companies(load_companies_from_json(args.companies))
    companies = store.list_companies()
    if args.company:
        companies = [c for c in companies if c.ticker.upper() == args.company.upper()]
        if not companies:
            raise SystemExit(f"Unknown company ticker: {args.company}")

    client = BlueskyClient()
    detector = MentionDetector()
    semaphore = asyncio.Semaphore(args.concurrency)

    search_tasks = []
    for company in companies:
        assert company.id is not None
        for query in _queries_for_company(company.ticker, company.name):
            search_tasks.append(
                _search_company_query(
                    client,
                    semaphore,
                    company_id=company.id,
                    query=query,
                    limit=args.limit,
                    lang=args.lang,
                )
            )

    results = await asyncio.gather(*search_tasks)

    documents_saved = 0
    mentions_saved = 0

    for result in results:
        for post in result.posts:
            document_id = store.save_raw_document(
                provider="bluesky",
                external_id=post.external_id,
                company_id=result.company_id,
                query=result.query,
                source_url=post.url,
                title=None,
                text=post.text,
                author=post.author_handle or post.author,
                published_at=post.published_at,
                metadata=post.metadata,
            )
            documents_saved += 1

            # Detect all tracked companies in the post, not only the query company.
            for mention in detector.detect(post.text, companies):
                assert mention.company.id is not None
                store.save_mention(
                    company_id=mention.company.id,
                    raw_document_id=document_id,
                    matched_text=mention.matched_text,
                    match_type=mention.match_type,
                    confidence=mention.confidence,
                )
                mentions_saved += 1

    store.close()
    print(f"Bluesky ingestion complete: documents={documents_saved}, mentions={mentions_saved}")


async def _search_company_query(
    client: BlueskyClient,
    semaphore: asyncio.Semaphore,
    *,
    company_id: int,
    query: str,
    limit: int,
    lang: str | None,
) -> SearchResult:
    async with semaphore:
        posts, _cursor = await client.search_posts(query, limit=limit, lang=lang)
    return SearchResult(company_id=company_id, query=query, posts=posts)


def _queries_for_company(ticker: str, name: str) -> list[str]:
    return [f"${ticker}", ticker, f"{name} stock"]


if __name__ == "__main__":
    main()
