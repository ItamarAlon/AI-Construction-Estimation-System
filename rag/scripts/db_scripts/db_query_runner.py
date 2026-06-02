from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv


class WriteQueryExecutionError(RuntimeError):
    """Raised when executing or committing a write SQL query fails."""


def _load_read_db_config() -> dict[str, Any]:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    return {
        "host": os.getenv("PG_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_PORT", "5435")),
        "user": os.getenv("PG_READ_USER", os.getenv("PG_USER", "receipt_user")),
        "password": os.getenv(
            "PG_READ_PASSWORD", os.getenv("PG_PASSWORD", "receipt_pass")
        ),
        "dbname": os.getenv("PG_DATABASE", "receipt_db"),
    }


def _load_write_db_config() -> dict[str, Any]:
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    return {
        "host": os.getenv("PG_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_PORT", "5435")),
        "user": os.getenv("PG_WRITE_USER", os.getenv("PG_USER", "receipt_user")),
        "password": os.getenv(
            "PG_WRITE_PASSWORD", os.getenv("PG_PASSWORD", "receipt_pass")
        ),
        "dbname": os.getenv("PG_DATABASE", "receipt_db"),
    }


def _normalize_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    return query.strip().rstrip(";")


def _contains_write_keyword(query: str) -> bool:
    return bool(re.search(r"\b(insert|update|delete)\b", query.lower()))


def run_read_query(query: str) -> list[dict[str, Any]]:
    """
    Runs a read-only SQL query using the read-role credentials.
    Returns result rows as dicts.
    """
    print(f"query: {query}")
    normalized = _normalize_query(query)
    if not normalized.lower().startswith(("select", "with")):
        raise ValueError("Read query must start with SELECT or WITH.")
    if normalized.lower().startswith("with") and _contains_write_keyword(normalized):
        raise ValueError("Read query cannot include INSERT, UPDATE, or DELETE statements.")

    with psycopg.connect(**_load_read_db_config(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(normalized)
            return list(cur.fetchall())


def run_write_query(
    query: str, params: dict[str, Any] | tuple[Any, ...] | None = None
) -> int:
    """
    Runs a write SQL query using the write-role credentials.
    Returns number of affected rows.
    """
    normalized = _normalize_query(query)
    lower = normalized.lower()
    if lower.startswith("select"):
        raise ValueError("Write query cannot start with SELECT.")
    if lower.startswith("with") and not _contains_write_keyword(normalized):
        raise ValueError(
            "Write CTE query must include INSERT, UPDATE, or DELETE statements."
        )
    if not lower.startswith("with") and not _contains_write_keyword(normalized):
        raise ValueError(
            "Write query must include INSERT, UPDATE, or DELETE statements."
        )

    try:
        with psycopg.connect(**_load_write_db_config()) as conn:
            with conn.cursor() as cur:
                if params is None:
                    cur.execute(normalized)
                else:
                    cur.execute(normalized, params)
                affected_rows = cur.rowcount if cur.rowcount is not None else 0
            conn.commit()
            return affected_rows
    except psycopg.Error as exc:
        raise WriteQueryExecutionError(f"Write query failed: {exc}") from exc
