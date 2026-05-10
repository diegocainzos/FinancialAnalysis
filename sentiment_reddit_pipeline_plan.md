# Reddit Sentiment Analysis Pipeline Plan

## Goal

Build a modular pipeline that extracts Reddit data about companies/tickers, stores raw and processed data in a database, runs sentiment analysis, and exposes clean sentiment signals for a future investment-tracking application.

## High-level Architecture

```txt
Company Watchlist
  ↓
Reddit Ingestion Service
  ↓
Raw Data Store
  ↓
NLP Processing Pipeline
  ↓
Processed Analytics Store
  ↓
Future API / Application Integration
```

## Main Modules

### 1. Company Watchlist

Stores the companies to track.

Example:

```json
[
  {
    "name": "Apple",
    "ticker": "AAPL",
    "aliases": ["Apple", "$AAPL", "iPhone", "Mac"]
  },
  {
    "name": "Tesla",
    "ticker": "TSLA",
    "aliases": ["Tesla", "$TSLA", "Elon", "Cybertruck"]
  }
]
```

Important: company matching needs care because names like Apple, Meta, Amazon, and Ford can be ambiguous.

### 2. Reddit Ingestion Service

Collects posts and comments from Reddit.

Possible sources:

- `r/stocks`
- `r/investing`
- `r/wallstreetbets`
- `r/SecurityAnalysis`
- `r/options`
- `r/ValueInvesting`

Collected fields:

- Reddit ID
- title
- body/text
- subreddit
- author
- score/upvotes
- comment count
- creation timestamp
- permalink
- ingestion timestamp

The ingestion service should separate:

1. Discovery: find relevant posts.
2. Expansion: fetch comments for relevant posts.
3. Persistence: save raw data.

### 3. Database

Recommended for the final project: **PostgreSQL**.

Suggested tables:

- `companies`
- `reddit_posts`
- `reddit_comments`
- `company_mentions`
- `sentiment_results`
- `sentiment_aggregates`
- `ingestion_runs`

### 4. NLP Processing Pipeline

Pipeline stages:

1. Clean Reddit text.
2. Detect company/ticker mentions.
3. Run sentiment analysis.
4. Save sentiment results.
5. Aggregate sentiment by company and time period.

Recommended first model: **FinBERT**, because it is trained for financial text.

Possible models:

- `ProsusAI/finbert`
- `yiyanghkust/finbert-tone`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

### 5. Aggregation

Aggregate by:

- company
- hour/day
- subreddit
- source type: post or comment

Useful metrics:

- mention count
- positive count
- negative count
- neutral count
- average sentiment
- weighted sentiment
- top positive/negative examples

## Recommended MVP

### Stack

- Python
- Reddit API via PRAW or direct HTTP OAuth2
- PostgreSQL
- SQLAlchemy
- Alembic
- Hugging Face Transformers
- FinBERT
- Docker Compose

### MVP Flow

```txt
Load companies
→ fetch Reddit posts/comments
→ save raw data
→ detect company mentions
→ run sentiment
→ save sentiment results
→ aggregate by company/day
```

### Initial Scope

Companies:

- Apple / AAPL
- Tesla / TSLA
- Nvidia / NVDA
- Microsoft / MSFT
- Amazon / AMZN

Subreddits:

- `r/stocks`
- `r/investing`
- `r/wallstreetbets`

Pipeline mode:

- Batch ingestion first.
- Example: run every hour or manually during development.

## Proposed Project Structure

```txt
project/
├── config/
│   └── companies.yaml
├── ingestion/
│   └── reddit_client.py
├── storage/
│   ├── models.py
│   └── repository.py
├── nlp/
│   ├── cleaner.py
│   ├── mention_detector.py
│   └── sentiment_analyzer.py
├── pipeline/
│   └── run_pipeline.py
└── api/
    └── future FastAPI integration
```

## Architecture Decision

Start with a **batch pipeline**, not real-time streaming.

Reasons:

- Easier to implement.
- Easier to debug.
- Better for Reddit API rate limits.
- Enough for investment sentiment tracking.
- Easier to integrate later into a larger application.

## Open Decisions

Before implementation, decide:

1. PostgreSQL or SQLite for the first prototype.
2. Posts only or posts + comments.
3. Ticker-only matching or ticker + aliases.
4. Direct Reddit API calls or PRAW wrapper.
5. Manual CLI pipeline first or scheduled jobs from the start.
