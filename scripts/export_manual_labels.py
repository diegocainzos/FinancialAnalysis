#!/usr/bin/env python3
"""Export FinBERT predictions to CSV for manual labeling."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rows for manual sentiment labeling")
    parser.add_argument("--db", default="data/sentiment.db", help="SQLite database path")
    parser.add_argument(
        "--model-name",
        default="finbert:ProsusAI/finbert",
        help="Model name to export predictions from",
    )
    parser.add_argument(
        "--output",
        default="data/manual_labeling_finbert.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        select
            sr.id as sentiment_result_id,
            sr.raw_document_id,
            c.ticker,
            d.provider,
            d.text,
            sr.sentiment_label as finbert_label,
            sr.sentiment_score as finbert_score,
            sr.confidence as finbert_confidence,
            '' as manual_label,
            '' as notes
        from sentiment_results sr
        join raw_documents d on d.id = sr.raw_document_id
        join companies c on c.id = sr.company_id
        where sr.model_name = ?
        order by sr.id
        """,
        (args.model_name,),
    ).fetchall()

    conn.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sentiment_result_id",
        "raw_document_id",
        "ticker",
        "provider",
        "text",
        "finbert_label",
        "finbert_score",
        "finbert_confidence",
        "manual_label",
        "notes",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"Exported {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
