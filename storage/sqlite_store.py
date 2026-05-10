"""SQLite persistence for provider-based sentiment ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Company:
    id: int | None
    name: str
    ticker: str
    aliases: list[str]


@dataclass(frozen=True)
class PendingSentimentDocument:
    raw_document_id: int
    company_id: int
    provider: str
    query: str
    text: str
    published_at: str | None


class SQLiteStore:
    def __init__(self, db_path: str | Path = "data/sentiment.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists companies (
                id integer primary key autoincrement,
                name text not null,
                ticker text not null unique,
                aliases_json text not null,
                created_at text not null
            );

            create table if not exists raw_documents (
                id integer primary key autoincrement,
                provider text not null,
                external_id text not null,
                company_id integer,
                query text not null,
                source_url text,
                title text,
                text text not null,
                author text,
                published_at text,
                ingested_at text not null,
                metadata_json text not null,
                unique(provider, external_id),
                foreign key(company_id) references companies(id)
            );

            create table if not exists company_mentions (
                id integer primary key autoincrement,
                company_id integer not null,
                raw_document_id integer not null,
                matched_text text not null,
                match_type text not null,
                confidence real not null,
                created_at text not null,
                unique(company_id, raw_document_id, matched_text),
                foreign key(company_id) references companies(id),
                foreign key(raw_document_id) references raw_documents(id)
            );

            create table if not exists sentiment_results (
                id integer primary key autoincrement,
                company_id integer not null,
                raw_document_id integer not null,
                sentiment_label text not null,
                sentiment_score real not null,
                confidence real not null,
                model_name text not null,
                processed_at text not null,
                unique(company_id, raw_document_id, model_name),
                foreign key(company_id) references companies(id),
                foreign key(raw_document_id) references raw_documents(id)
            );

            create index if not exists idx_raw_documents_provider on raw_documents(provider);
            create index if not exists idx_raw_documents_company on raw_documents(company_id);
            create index if not exists idx_raw_documents_published on raw_documents(published_at);
            create index if not exists idx_sentiment_company on sentiment_results(company_id);
            create index if not exists idx_sentiment_document on sentiment_results(raw_document_id);
            """
        )
        self.conn.commit()

    def upsert_companies(self, companies: Iterable[Company]) -> None:
        now = _dt(datetime.now(timezone.utc))
        self.conn.executemany(
            """
            insert into companies(name, ticker, aliases_json, created_at)
            values (?, ?, ?, ?)
            on conflict(ticker) do update set
                name = excluded.name,
                aliases_json = excluded.aliases_json
            """,
            [(c.name, c.ticker, json.dumps(c.aliases), now) for c in companies],
        )
        self.conn.commit()

    def list_companies(self) -> list[Company]:
        rows = self.conn.execute("select * from companies order by ticker").fetchall()
        return [
            Company(
                id=row["id"],
                name=row["name"],
                ticker=row["ticker"],
                aliases=json.loads(row["aliases_json"]),
            )
            for row in rows
        ]

    def save_raw_document(
        self,
        *,
        provider: str,
        external_id: str,
        company_id: int | None,
        query: str,
        source_url: str | None,
        title: str | None,
        text: str,
        author: str | None,
        published_at: datetime | None,
        metadata: dict[str, Any],
    ) -> int:
        now = _dt(datetime.now(timezone.utc))
        self.conn.execute(
            """
            insert into raw_documents(
                provider, external_id, company_id, query, source_url, title, text,
                author, published_at, ingested_at, metadata_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(provider, external_id) do update set
                company_id = coalesce(raw_documents.company_id, excluded.company_id),
                query = excluded.query,
                source_url = excluded.source_url,
                text = excluded.text,
                author = excluded.author,
                published_at = excluded.published_at,
                metadata_json = excluded.metadata_json
            """,
            (
                provider,
                external_id,
                company_id,
                query,
                source_url,
                title,
                text,
                author,
                _dt(published_at),
                now,
                json.dumps(metadata),
            ),
        )
        row = self.conn.execute(
            "select id from raw_documents where provider = ? and external_id = ?",
            (provider, external_id),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def save_mention(
        self,
        *,
        company_id: int,
        raw_document_id: int,
        matched_text: str,
        match_type: str,
        confidence: float,
    ) -> None:
        self.conn.execute(
            """
            insert or ignore into company_mentions(
                company_id, raw_document_id, matched_text, match_type, confidence, created_at
            ) values (?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                raw_document_id,
                matched_text,
                match_type,
                confidence,
                _dt(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def list_documents_pending_sentiment(
        self,
        *,
        model_name: str,
        limit: int = 100,
    ) -> list[PendingSentimentDocument]:
        rows = self.conn.execute(
            """
            select distinct
                d.id as raw_document_id,
                m.company_id,
                d.provider,
                d.query,
                d.text,
                d.published_at
            from company_mentions m
            join raw_documents d on d.id = m.raw_document_id
            left join sentiment_results s
                on s.raw_document_id = m.raw_document_id
                and s.company_id = m.company_id
                and s.model_name = ?
            where s.id is null
            order by d.published_at desc, d.id desc
            limit ?
            """,
            (model_name, limit),
        ).fetchall()
        return [
            PendingSentimentDocument(
                raw_document_id=row["raw_document_id"],
                company_id=row["company_id"],
                provider=row["provider"],
                query=row["query"],
                text=row["text"],
                published_at=row["published_at"],
            )
            for row in rows
        ]

    def save_sentiment_result(
        self,
        *,
        company_id: int,
        raw_document_id: int,
        sentiment_label: str,
        sentiment_score: float,
        confidence: float,
        model_name: str,
    ) -> None:
        self.conn.execute(
            """
            insert into sentiment_results(
                company_id, raw_document_id, sentiment_label, sentiment_score,
                confidence, model_name, processed_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            on conflict(company_id, raw_document_id, model_name) do update set
                sentiment_label = excluded.sentiment_label,
                sentiment_score = excluded.sentiment_score,
                confidence = excluded.confidence,
                processed_at = excluded.processed_at
            """,
            (
                company_id,
                raw_document_id,
                sentiment_label,
                sentiment_score,
                confidence,
                model_name,
                _dt(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()


def load_companies_from_json(path: str | Path) -> list[Company]:
    data = json.loads(Path(path).read_text())
    return [
        Company(
            id=None,
            name=item["name"],
            ticker=item["ticker"],
            aliases=list(item.get("aliases", [])),
        )
        for item in data
    ]


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()
