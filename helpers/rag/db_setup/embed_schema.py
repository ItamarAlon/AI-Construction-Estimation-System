from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

__all__ = ["get_schema_retriever"]


@dataclass
class SchemaDoc:
    schema_name: str
    object_type: str
    object_name: str
    content: str
    metadata: dict


@dataclass
class SchemaMatch:
    schema_name: str
    object_type: str
    object_name: str
    content: str
    metadata: dict
    similarity: float


class SchemaRetriever:
    """
    Lab-style schema retriever over schema_embeddings.

    - retrieve(): nearest-neighbor schema chunks for a natural-language question
    - retrieve_context(): concatenated schema context string for LLM SQL generation
    """

    def __init__(self, cfg: dict, default_k: int = 5) -> None:
        self.cfg = cfg
        self.default_k = default_k
        self.client = create_embedding_client(cfg)
        self.embedding_model = cfg["embedding_model"]

    def _embed(self, text: str) -> list[float]:
        return self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        ).data[0].embedding

    def retrieve(self, question: str, k: int | None = None) -> list[SchemaMatch]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        limit = k if k is not None else self.default_k
        query_embedding = self._embed(question.strip())
        query_embedding_literal = to_vector_literal(query_embedding)
        rows: list[tuple[Any, ...]]

        with psycopg.connect(**self.cfg["db"]) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        schema_name,
                        object_type,
                        object_name,
                        content,
                        COALESCE(metadata, '{}'::jsonb) AS metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM schema_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding_literal, query_embedding_literal, limit),
                )
                rows = cur.fetchall()

        return [
            SchemaMatch(
                schema_name=row[0],
                object_type=row[1],
                object_name=row[2],
                content=row[3],
                metadata=row[4] if isinstance(row[4], dict) else {},
                similarity=float(row[5]),
            )
            for row in rows
        ]

    def retrieve_context(self, question: str, k: int | None = None) -> str:
        matches = self.retrieve(question, k)
        if not matches:
            return "No schema context found. Run scripts/embed_schema.py first."
        return "\n\n---\n\n".join(match.content for match in matches)

    def invoke(self, question: str) -> list[SchemaMatch]:
        """
        Mirrors retriever.invoke(question) style from the JS lab.
        """
        return self.retrieve(question)


def load_config() -> dict:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    api_key = openrouter_api_key or openai_api_key
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment."
        )

    use_openrouter = bool(openrouter_api_key)
    default_embedding_model = (
        "openai/text-embedding-3-small" if use_openrouter else "text-embedding-3-small"
    )

    return {
        "db": {
            "host": os.getenv("PG_HOST", "127.0.0.1"),
            "port": int(os.getenv("PG_PORT", "5435")),
            "user": os.getenv("PG_EMBED_USER", os.getenv("PG_USER", "receipt_user")),
            "password": os.getenv(
                "PG_EMBED_PASSWORD", os.getenv("PG_PASSWORD", "receipt_pass")
            ),
            "dbname": os.getenv("PG_DATABASE", "receipt_db"),
        },
        "embedding_model": os.getenv("EMBEDDING_MODEL", default_embedding_model),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "1536")),
        "api_key": api_key,
        "use_openrouter": use_openrouter,
    }


def create_embedding_client(cfg: dict) -> OpenAI:
    if cfg["use_openrouter"]:
        return OpenAI(
            api_key=cfg["api_key"],
            base_url="https://openrouter.ai/api/v1",
        )
    return OpenAI(api_key=cfg["api_key"])


def build_schema_docs(conn: psycopg.Connection) -> list[SchemaDoc]:
    docs: list[SchemaDoc] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        tables = cur.fetchall()

        for schema_name, table_name in tables:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (schema_name, table_name),
            )
            columns = cur.fetchall()

            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = %s AND tablename = %s
                ORDER BY indexname;
                """,
                (schema_name, table_name),
            )
            indexes = cur.fetchall()

            col_parts = []
            for column_name, data_type, is_nullable, column_default in columns:
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                default = f" DEFAULT {column_default}" if column_default else ""
                col_parts.append(f"{column_name} {data_type} {nullable}{default}")

            index_names = [name for name, _ in indexes]
            content = (
                f"Table {schema_name}.{table_name}\n"
                f"Columns:\n- " + "\n- ".join(col_parts) + "\n"
                f"Indexes: {', '.join(index_names) if index_names else 'none'}"
            )

            docs.append(
                SchemaDoc(
                    schema_name=schema_name,
                    object_type="table",
                    object_name=table_name,
                    content=content,
                    metadata={
                        "columns": [c[0] for c in columns],
                        "indexes": index_names,
                    },
                )
            )

            for column_name, data_type, is_nullable, column_default in columns:
                default = f", default={column_default}" if column_default else ""
                column_content = (
                    f"Column {schema_name}.{table_name}.{column_name}: "
                    f"type={data_type}, nullable={is_nullable}{default}"
                )
                docs.append(
                    SchemaDoc(
                        schema_name=schema_name,
                        object_type="column",
                        object_name=f"{table_name}.{column_name}",
                        content=column_content,
                        metadata={
                            "table": table_name,
                            "column": column_name,
                            "data_type": data_type,
                            "is_nullable": is_nullable,
                        },
                    )
                )

    return docs


def embed_text(client: OpenAI, model: str, text: str) -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def get_schema_retriever(k: int = 8) -> SchemaRetriever:
    cfg = load_config()
    return SchemaRetriever(cfg=cfg, default_k=k)


def main() -> None:
    cfg = load_config()
    client = create_embedding_client(cfg)

    with psycopg.connect(**cfg["db"]) as conn:
        register_vector(conn)
        docs = build_schema_docs(conn)

        print(f"Embedding {len(docs)} schema docs...")
        with conn.cursor() as cur:
            for doc in docs:
                emb = embed_text(client, cfg["embedding_model"], doc.content)
                if len(emb) != cfg["embedding_dim"]:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {cfg['embedding_dim']}, got {len(emb)}"
                    )

                cur.execute(
                    """
                    DELETE FROM schema_embeddings
                    WHERE schema_name = %s
                      AND object_type = %s
                      AND object_name = %s;
                    """,
                    (doc.schema_name, doc.object_type, doc.object_name),
                )

                cur.execute(
                    """
                    INSERT INTO schema_embeddings
                        (schema_name, object_type, object_name, content_hash, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s);
                    """,
                    (
                        doc.schema_name,
                        doc.object_type,
                        doc.object_name,
                        content_hash(doc.content),
                        doc.content,
                        json.dumps(doc.metadata),
                        emb,
                    ),
                )
        conn.commit()

    print("Schema embeddings updated.")
    print("Retriever ready: call get_schema_retriever() from this module.")


if __name__ == "__main__":
    main()
