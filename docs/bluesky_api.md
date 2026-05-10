# Bluesky API Notes for the Sentiment Pipeline

Bluesky uses the AT Protocol. For this project we can start with the public AppView endpoint:

```txt
https://api.bsky.app
```

## Endpoint used in the MVP

### Search public posts

```http
GET /xrpc/app.bsky.feed.searchPosts
```

Example:

```http
GET https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=%24TSLA&limit=25
```

Useful parameters:

- `q`: search query.
- `limit`: max posts, usually up to 100.
- `cursor`: pagination cursor returned by the previous response.
- `lang`: optional language filter, e.g. `en`, `es`.

Returned fields we store:

- `uri`: stable AT Protocol post URI.
- `cid`: content identifier.
- `record.text`: post text.
- `record.createdAt`: published timestamp.
- `author.handle`: author handle.
- engagement and other raw fields inside `metadata_json`.

## MVP query strategy

For each company:

```txt
$TICKER
TICKER
Company stock
```

Examples:

```txt
$TSLA
TSLA
Tesla stock
$NVDA
NVDA
Nvidia stock
```

## Current implementation

CLI:

```bash
python3 -m pipeline.ingest_bluesky --limit 25
python3 -m pipeline.ingest_bluesky --company TSLA --limit 50 --lang en
python3 -m pipeline.ingest_bluesky --company TSLA --limit 50 --lang en --concurrency 3
```

Database:

```txt
data/sentiment.db
```

Tables:

- `companies`
- `raw_documents`
- `company_mentions`

## Next steps

1. Add sentiment processing table: `sentiment_results`.
2. Add aggregate table: `sentiment_aggregates`.
3. Add FinBERT or a smaller model depending on local resources.
4. Add scheduled ingestion.
5. Add a FastAPI read API for the future app.
