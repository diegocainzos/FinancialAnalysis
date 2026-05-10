"""Minimal Bluesky/AT Protocol search client.

Uses the public AppView endpoint, so no credentials are required for basic public
post search. This is enough for the first ingestion MVP.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BlueskyPost:
    external_id: str
    uri: str
    cid: str
    text: str
    author: str | None
    author_handle: str | None
    published_at: datetime | None
    url: str | None
    metadata: dict[str, Any]


class BlueskyClient:
    """Client for `app.bsky.feed.searchPosts`.

    API reference conceptually:
    GET /xrpc/app.bsky.feed.searchPosts?q=<query>&limit=<n>&cursor=<cursor>
    """

    def __init__(self, base_url: str = "https://api.bsky.app") -> None:
        self.base_url = base_url.rstrip("/")

    async def search_posts(
        self,
        query: str,
        *,
        limit: int = 25,
        cursor: str | None = None,
        lang: str | None = None,
    ) -> tuple[list[BlueskyPost], str | None]:
        """Search posts asynchronously.

        The standard library has no native async HTTP client, so this keeps the
        project dependency-free by running the blocking request in a worker
        thread. The pipeline can still execute many searches concurrently.
        """
        payload = await asyncio.to_thread(
            self._get_search_payload,
            query,
            limit=limit,
            cursor=cursor,
            lang=lang,
        )
        posts = [self._parse_post(item) for item in payload.get("posts", [])]
        return posts, payload.get("cursor")

    def _get_search_payload(
        self,
        query: str,
        *,
        limit: int,
        cursor: str | None,
        lang: str | None,
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {"q": query, "limit": min(limit, 100)}
        if cursor:
            params["cursor"] = cursor
        if lang:
            params["lang"] = lang

        url = f"{self.base_url}/xrpc/app.bsky.feed.searchPosts?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "company-sentiment-pipeline/0.1"})

        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_post(self, item: dict[str, Any]) -> BlueskyPost:
        record = item.get("record") or {}
        author = item.get("author") or {}
        uri = item.get("uri", "")
        cid = item.get("cid", "")
        handle = author.get("handle")
        created_at = _parse_datetime(record.get("createdAt"))

        return BlueskyPost(
            external_id=uri or cid,
            uri=uri,
            cid=cid,
            text=record.get("text") or "",
            author=author.get("displayName"),
            author_handle=handle,
            published_at=created_at,
            url=_post_url(handle, uri),
            metadata=item,
        )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _post_url(handle: str | None, uri: str) -> str | None:
    # AT URI format: at://did:plc:.../app.bsky.feed.post/<rkey>
    if not handle or not uri:
        return None
    rkey = uri.rstrip("/").split("/")[-1]
    return f"https://bsky.app/profile/{handle}/post/{rkey}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
