# Multi-Provider Company Sentiment Pipeline Plan

## Goal

Build a provider-agnostic company sentiment pipeline that collects public market-related text, stores raw and normalized data, detects company mentions, runs sentiment analysis, aggregates results, and exposes the results for a future investment-tracking application.

## Current Implementation Status

Implemented MVP pieces:

- **Bluesky async ingestion**
  - File: `ingestion/bluesky_client.py`
  - Uses AT Protocol AppView endpoint: `https://api.bsky.app/xrpc/app.bsky.feed.searchPosts`
  - Public post search works without credentials.
  - `search_posts()` is asynchronous and uses `asyncio.to_thread()` around stdlib `urllib` to keep the project dependency-free.

- **Concurrent ingestion CLI**
  - File: `pipeline/ingest_bluesky.py`
  - Uses `asyncio.run()`, `asyncio.gather()`, and a semaphore-controlled `--concurrency` option.
  - Searches each company using three query patterns: `$TICKER`, `TICKER`, and `Company stock`.
  - Saves raw posts and detected mentions to SQLite.

- **SQLite storage**
  - File: `storage/sqlite_store.py`
  - Current DB: `data/sentiment.db`
  - Current tables: `companies`, `raw_documents`, `company_mentions`.
  - Deduplicates raw posts with unique key `(provider, external_id)`.
  - Deduplicates mentions with unique key `(company_id, raw_document_id, matched_text)`.

- **Rule-based mention detection**
  - File: `nlp/mention_detector.py`
  - Detects tickers, cashtags, company names, and aliases.
  - Current confidence heuristics:
    - `cashtag`: 0.95
    - `ticker`: 0.85
    - `company_name`: 0.75
    - `alias`: 0.60

- **Company watchlist**
  - File: `config/companies.json`
  - Current companies: AAPL, TSLA, NVDA, MSFT, AMZN.

- **Sentiment processing MVP**
  - Files: `nlp/sentiment_analyzer.py`, `nlp/cleaner.py`, `pipeline/process_sentiment.py`
  - Default analyzer: VADER if `vaderSentiment` is installed; otherwise a deterministic local finance/social lexicon fallback.
  - Saves per-company/per-document results into `sentiment_results`.
  - CLI command: `python3 -m pipeline.process_sentiment --limit 100`

- **Tests**
  - File: `tests/test_pipeline_modules.py`
  - Covers mention detection, SQLite upsert/idempotency, sentiment persistence, text cleaning, sentiment labels, and async Bluesky payload parsing without network.
  - Test command: `python3 -m unittest discover -s tests -v`

- **Bluesky API notes**
  - File: `docs/bluesky_api.md`

## Target Architecture

```txt
config/companies.json
  ↓
Provider ingestion layer
  ├── BlueskyProvider / BlueskyClient          implemented
  ├── TavilyProvider                           planned
  ├── News/RSS provider                        optional
  └── RedditProvider                           deferred/unavailable
  ↓
Raw document storage
  ↓
Company mention detection
  ↓
Text cleaning + sentiment model
  ↓
Sentiment result storage
  ↓
Time/company/provider aggregates
  ↓
Future API / dashboard / investment app
```

## Current Data Model

### `companies`

Stores the watchlist.

```txt
id
name
ticker
aliases_json
created_at
```

### `raw_documents`

Provider-agnostic raw content table.

```txt
id
provider              -- e.g. bluesky, tavily, news
external_id           -- e.g. AT URI, URL hash, provider ID
company_id            -- company associated with ingestion query
query                 -- search query used
source_url
title
text
author
published_at
ingested_at
metadata_json         -- raw provider payload
```

### `company_mentions`

Links any raw document to one or more detected companies.

```txt
id
company_id
raw_document_id
matched_text
match_type            -- cashtag, ticker, company_name, alias
confidence
created_at
```

### `sentiment_results`

Per-document/per-company sentiment output.

```txt
id
company_id
raw_document_id
sentiment_label       -- positive, negative, neutral
sentiment_score       -- normalized score, e.g. -1.0 to +1.0
confidence
model_name
processed_at
```

## Planned Data Model Extensions

### `sentiment_aggregates`

Aggregated sentiment by company, provider, and period.

```txt
id
company_id
provider
period_start
period_end
period_type           -- hourly, daily
mention_count
positive_count
negative_count
neutral_count
average_score
weighted_score
created_at
```

### `ingestion_runs`

Operational tracking for scheduled jobs.

```txt
id
provider
started_at
finished_at
status
query_count
documents_seen
documents_saved
error_message
```

## Commands

### Run Bluesky ingestion for all companies

```bash
python3 -m pipeline.ingest_bluesky --limit 25
```

### Run Bluesky ingestion for one company

```bash
python3 -m pipeline.ingest_bluesky --company TSLA --limit 50 --lang en
```

### Run with explicit concurrency

```bash
python3 -m pipeline.ingest_bluesky --company NVDA --limit 50 --lang en --concurrency 3
```

### Process pending sentiment

```bash
python3 -m pipeline.process_sentiment --limit 100
```

### Run tests

```bash
python3 -m unittest discover -s tests -v
```

### Compile/check syntax

```bash
python3 -m compileall ingestion storage nlp pipeline
```

### Inspect SQLite manually

```bash
sqlite3 data/sentiment.db
```

Useful queries:

```sql
select provider, query, count(*) as docs
from raw_documents
group by provider, query;

select c.ticker, m.matched_text, m.match_type, m.confidence, substr(d.text, 1, 80) as preview
from company_mentions m
join companies c on c.id = m.company_id
join raw_documents d on d.id = m.raw_document_id
order by m.id desc
limit 20;
```

## Next Milestones

1. **Add aggregate builder**
   - File: `pipeline/build_aggregates.py`
   - Aggregate sentiment by company, provider, and day first; hourly later.
   - Acceptance: `sentiment_aggregates` contains daily counts and average/weighted sentiment.

2. **Improve provider abstraction**
   - New file: `ingestion/base.py`
   - Define a common provider interface so Bluesky, Tavily, news, or future Reddit ingestion can share the same pipeline.
   - Acceptance: provider-specific clients return a normalized document object.

3. **Add Tavily provider**
   - New file: `ingestion/tavily_client.py`
   - Use Tavily for news/web/company context, complementary to Bluesky social sentiment.
   - Acceptance: Tavily results save into `raw_documents` with `provider='tavily'`.

4. **Add scheduling**
   - Options: cron, APScheduler, or a simple periodic CLI wrapper.
   - Acceptance: ingestion and sentiment processing can run automatically on a configured interval.

5. **Expose read API**
   - New app layer, likely FastAPI.
   - Endpoints to add later:
     - `GET /companies`
     - `GET /companies/{ticker}/sentiment`
     - `GET /companies/{ticker}/documents`
     - `GET /sentiment/compare?tickers=TSLA,NVDA,MSFT`
   - Acceptance: future frontend/dashboard can query sentiment trends.

## Risks and Decisions Needed

- **Sentiment model choice**: FinBERT is finance-aware but may not be ideal for short, informal social posts. Need explicit model comparison before locking it in.
- **False positives in mention detection**: Names like Apple, Amazon, Meta, Ford, and aliases like Mac can be ambiguous.
- **Bluesky search limits/rate limits**: Current implementation lacks retry/backoff and cursor pagination beyond the first page.
- **SQLite concurrency**: Current approach fetches concurrently but writes synchronously, which is safe for MVP. PostgreSQL should replace SQLite for multi-worker production use.
- **Provider bias**: Bluesky may not represent investor sentiment broadly. Tavily/news sources should be added for balance.
- **Historical coverage**: Bluesky search may not provide complete historical data; aggregation should record ingestion time and published time.

## Recommended Immediate Next Step

Build daily sentiment aggregates:

1. Add `sentiment_aggregates` schema.
2. Create `pipeline/build_aggregates.py`.
3. Aggregate by company, provider, and day.
4. Add tests for aggregate counts and weighted score.

After that, add optional FinBERT support behind the existing `SentimentAnalyzer` interface.
