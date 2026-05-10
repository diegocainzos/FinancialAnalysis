from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ingestion.bluesky_client import BlueskyClient
from nlp.cleaner import clean_text
from nlp.mention_detector import MentionDetector
from nlp.sentiment_analyzer import VaderSentimentAnalyzer
from storage.sqlite_store import Company, SQLiteStore


class MentionDetectorTests(unittest.TestCase):
    def test_detects_ticker_cashtag_company_name_and_alias(self) -> None:
        companies = [
            Company(
                id=1,
                name="Tesla",
                ticker="TSLA",
                aliases=["Tesla", "$TSLA", "TSLA", "Cybertruck"],
            )
        ]

        mentions = MentionDetector().detect(
            "Tesla is moving again. $TSLA bulls mention Cybertruck demand.",
            companies,
        )

        by_text = {mention.matched_text: mention for mention in mentions}
        self.assertEqual(by_text["Tesla"].match_type, "company_name")
        self.assertEqual(by_text["$TSLA"].match_type, "cashtag")
        self.assertEqual(by_text["Cybertruck"].match_type, "alias")
        self.assertEqual(by_text["$TSLA"].confidence, 0.95)

    def test_does_not_match_ticker_inside_larger_word(self) -> None:
        companies = [Company(id=1, name="Tesla", ticker="TSLA", aliases=[])]

        mentions = MentionDetector().detect("This text says XTSLAY, not the ticker.", companies)

        self.assertEqual(mentions, [])

    def test_detects_multiple_companies_in_same_text(self) -> None:
        companies = [
            Company(id=1, name="Tesla", ticker="TSLA", aliases=["Tesla"]),
            Company(id=2, name="Nvidia", ticker="NVDA", aliases=["Nvidia", "NVIDIA"]),
        ]

        mentions = MentionDetector().detect("TSLA and NVIDIA are both AI trades.", companies)

        tickers = {mention.company.ticker for mention in mentions}
        self.assertEqual(tickers, {"TSLA", "NVDA"})


class SQLiteStoreTests(unittest.TestCase):
    def test_upserts_companies_and_raw_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStore(Path(tmpdir) / "test.db")
            store.init_schema()
            store.upsert_companies(
                [Company(id=None, name="Tesla", ticker="TSLA", aliases=["Tesla", "$TSLA"])]
            )

            company = store.list_companies()[0]
            self.assertIsNotNone(company.id)
            self.assertEqual(company.ticker, "TSLA")

            first_id = store.save_raw_document(
                provider="bluesky",
                external_id="at://example/post/1",
                company_id=company.id,
                query="TSLA",
                source_url="https://bsky.app/profile/example/post/1",
                title=None,
                text="Original text about TSLA",
                author="example.bsky.social",
                published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                metadata={"likeCount": 1},
            )
            second_id = store.save_raw_document(
                provider="bluesky",
                external_id="at://example/post/1",
                company_id=company.id,
                query="TSLA",
                source_url="https://bsky.app/profile/example/post/1",
                title=None,
                text="Updated text about TSLA",
                author="example.bsky.social",
                published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                metadata={"likeCount": 2},
            )

            self.assertEqual(first_id, second_id)
            rows = store.conn.execute("select text, metadata_json from raw_documents").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "Updated text about TSLA")
            self.assertIn('"likeCount": 2', rows[0]["metadata_json"])
            store.close()

    def test_save_mention_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStore(Path(tmpdir) / "test.db")
            store.init_schema()
            store.upsert_companies([Company(id=None, name="Nvidia", ticker="NVDA", aliases=[])])
            company = store.list_companies()[0]
            assert company.id is not None
            document_id = store.save_raw_document(
                provider="bluesky",
                external_id="post-1",
                company_id=company.id,
                query="NVDA",
                source_url=None,
                title=None,
                text="NVDA is up",
                author=None,
                published_at=None,
                metadata={},
            )

            for _ in range(2):
                store.save_mention(
                    company_id=company.id,
                    raw_document_id=document_id,
                    matched_text="NVDA",
                    match_type="ticker",
                    confidence=0.85,
                )

            count = store.conn.execute("select count(*) from company_mentions").fetchone()[0]
            self.assertEqual(count, 1)
            store.close()

    def test_lists_pending_sentiment_and_saves_result_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStore(Path(tmpdir) / "test.db")
            store.init_schema()
            store.upsert_companies([Company(id=None, name="Tesla", ticker="TSLA", aliases=[])])
            company = store.list_companies()[0]
            assert company.id is not None
            document_id = store.save_raw_document(
                provider="bluesky",
                external_id="post-2",
                company_id=company.id,
                query="TSLA",
                source_url=None,
                title=None,
                text="TSLA is up after strong growth",
                author=None,
                published_at=None,
                metadata={},
            )
            store.save_mention(
                company_id=company.id,
                raw_document_id=document_id,
                matched_text="TSLA",
                match_type="ticker",
                confidence=0.85,
            )

            pending = store.list_documents_pending_sentiment(model_name="test-model", limit=10)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].raw_document_id, document_id)

            for _ in range(2):
                store.save_sentiment_result(
                    company_id=company.id,
                    raw_document_id=document_id,
                    sentiment_label="positive",
                    sentiment_score=0.8,
                    confidence=0.8,
                    model_name="test-model",
                )

            pending_after = store.list_documents_pending_sentiment(model_name="test-model", limit=10)
            count = store.conn.execute("select count(*) from sentiment_results").fetchone()[0]
            self.assertEqual(pending_after, [])
            self.assertEqual(count, 1)
            store.close()


class SentimentAnalyzerTests(unittest.TestCase):
    def test_vader_fallback_labels_positive_negative_and_neutral(self) -> None:
        analyzer = VaderSentimentAnalyzer()

        positive = analyzer.analyze("TSLA is up after strong growth and beats estimates")
        negative = analyzer.analyze("NVDA is down after weak losses and downgrade risk")
        neutral = analyzer.analyze("MSFT reports a product update today")

        self.assertEqual(positive.label, "positive")
        self.assertGreater(positive.score, 0)
        self.assertEqual(negative.label, "negative")
        self.assertLess(negative.score, 0)
        self.assertEqual(neutral.label, "neutral")

    def test_clean_text_removes_urls_and_normalizes_whitespace(self) -> None:
        self.assertEqual(clean_text("TSLA   up\nhttps://example.com"), "TSLA up")


class BlueskyClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_posts_parses_api_payload_without_network(self) -> None:
        class FakeBlueskyClient(BlueskyClient):
            def _get_search_payload(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    "cursor": "next-page",
                    "posts": [
                        {
                            "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
                            "cid": "cid-123",
                            "record": {
                                "text": "NVDA is up after earnings",
                                "createdAt": "2026-01-01T12:00:00Z",
                            },
                            "author": {
                                "handle": "investor.bsky.social",
                                "displayName": "Investor",
                            },
                        }
                    ],
                }

        posts, cursor = await FakeBlueskyClient().search_posts("NVDA", limit=1)

        self.assertEqual(cursor, "next-page")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].text, "NVDA is up after earnings")
        self.assertEqual(posts[0].author_handle, "investor.bsky.social")
        self.assertEqual(
            posts[0].url,
            "https://bsky.app/profile/investor.bsky.social/post/xyz",
        )


if __name__ == "__main__":
    unittest.main()
