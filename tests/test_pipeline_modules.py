from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

from ingestion.bluesky_client import BlueskyClient
from nlp.cleaner import clean_text
from nlp.market_relevance import KeywordMarketRelevanceFilter, parse_hf_gguf_ref, parse_yes_no_response
from nlp.mention_detector import MentionDetector
from nlp.sentiment_analyzer import FinBertSentimentAnalyzer, VaderSentimentAnalyzer
from pipeline.ingest_bluesky import _queries_for_company
from pipeline.process_sentiment import build_analyzer
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


class MarketRelevanceTests(unittest.TestCase):
    def test_keyword_filter_accepts_market_text_and_rejects_product_text(self) -> None:
        relevance = KeywordMarketRelevanceFilter()

        self.assertTrue(
            relevance.is_market_relevant(
                ticker="AAPL",
                company_name="Apple",
                text="AAPL stock jumps after strong earnings and raised guidance",
            )
        )
        self.assertFalse(
            relevance.is_market_relevant(
                ticker="AAPL",
                company_name="Apple",
                text="I bought a new Apple Watch and I love it",
            )
        )

    def test_parse_hf_gguf_ref_rejects_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            parse_hf_gguf_ref("mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF")

    def test_parse_yes_no_response_handles_accents_and_case(self) -> None:
        self.assertTrue(parse_yes_no_response("SI"))
        self.assertTrue(parse_yes_no_response("Sí, relevante"))
        self.assertFalse(parse_yes_no_response("no"))
        self.assertTrue(parse_yes_no_response("quizas"))  # defaults to True now


class PipelineQueryTests(unittest.TestCase):
    def test_queries_for_company_are_market_oriented(self) -> None:
        queries = _queries_for_company("TSLA", "Tesla")
        self.assertIn("$TSLA", queries)
        self.assertIn("TSLA earnings", queries)
        self.assertIn("Tesla guidance", queries)
        self.assertNotIn("TSLA", queries)


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

    def test_finbert_maps_labels_to_pipeline_result_shape(self) -> None:
        outputs = [
            [{"label": "positive", "score": 0.97}],
            [{"label": "negative", "score": 0.88}],
            [{"label": "neutral", "score": 0.62}],
        ]

        def fake_classifier(text: str, **kwargs):  # type: ignore[no-untyped-def]
            return outputs.pop(0)

        analyzer = FinBertSentimentAnalyzer(classifier=fake_classifier)

        positive = analyzer.analyze("Strong guidance and better margins")
        negative = analyzer.analyze("Guidance cut and weaker demand")
        neutral = analyzer.analyze("Company held an investor event")

        self.assertEqual(positive.label, "positive")
        self.assertGreater(positive.score, 0)
        self.assertEqual(negative.label, "negative")
        self.assertLess(negative.score, 0)
        self.assertEqual(neutral.label, "neutral")
        self.assertEqual(neutral.score, 0.0)

    def test_process_pipeline_uses_finbert_by_default(self) -> None:
        with patch("pipeline.process_sentiment.FinBertSentimentAnalyzer") as finbert_cls:
            sentinel = object()
            finbert_cls.return_value = sentinel
            analyzer = build_analyzer(model="finbert")

        self.assertIs(analyzer, sentinel)
        finbert_cls.assert_called_once_with()

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
