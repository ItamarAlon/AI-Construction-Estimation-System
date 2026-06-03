from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

READ_ONLY_VERBS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "merge",
    "replace",
    "comment",
)


@dataclass
class ReadOnlySQLAgent:
    """
    Generates a PostgreSQL SQL query from a natural-language question and schema context.
    The agent enforces read-only SQL generation and validation (SELECT/WITH only).
    """

    model_name: str | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        api_key = openrouter_api_key or openai_api_key
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment.")

        use_openrouter = bool(openrouter_api_key)
        default_model = "openai/gpt-4o-mini" if use_openrouter else "gpt-4o-mini"
        self._llm = (
            ChatOpenAI(
                model=self.model_name or default_model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=self.temperature,
            )
            if use_openrouter
            else ChatOpenAI(
                model=self.model_name or default_model,
                api_key=api_key,
                temperature=self.temperature,
            )
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a PostgreSQL SQL assistant.\n"
                        "You must generate exactly one read-only SQL query.\n"
                        "Allowed query types: SELECT or WITH ... SELECT.\n"
                        "Disallowed operations: INSERT, UPDATE, DELETE, DROP, ALTER, "
                        "TRUNCATE, CREATE, GRANT, REVOKE, MERGE, REPLACE, COMMENT.\n"
                        "Never output explanations, markdown, or code fences.\n"
                        "Use only tables/columns that exist in the provided schema context.\n"
                        "Prefer explicit columns over SELECT * unless the question clearly asks for all columns.\n"
                        "For item-name/product-name filtering, prefer receipt_items.description.\n"
                        "Use receipt_items.metadata JSON fields only when the question explicitly asks for metadata.\n"
                        'Treat user words "store", "shop", and "vendor" as merchant references.\n'
                        "When users ask about a store, use merchants.name (via receipts.merchant_id join), "
                        "not receipts.store_name.\n"
                        "Interpret relative time phrases in questions using PostgreSQL date/time expressions.\n"
                        "Examples include today, yesterday, tomorrow, N days ago, last week, this week, next week, "
                        "last month, this month, next month, last year, and this year.\n"
                        "Use CURRENT_DATE/CURRENT_TIMESTAMP and date_trunc/interval arithmetic where appropriate, "
                        "and ensure date ranges are precise and inclusive/exclusive as needed.\n"
                        "For all text filters (for example category/store/vendor/item names), match "
                        "case-insensitively by default.\n"
                        "Prefer LOWER(column) = LOWER('value') for exact text matches, and ILIKE for partial matches.\n"
                        "Write user-provided text literals in lowercase unless exact original casing is explicitly required.\n"
                        "When phrasing is ambiguous between category and item text (for example 'receipts of food'), "
                        "prefer category intent first using receipts.category_id -> categories.name case-insensitive match.\n"
                        "Treat category synonyms and phrasing such as category, type, kind, class, group, "
                        "'of type X', and 'type X' as category intent.\n"
                        "For category-intent phrasing, filter via receipts.category_id joined to categories.name, "
                        "not via receipt_items.description.\n"
                        "Treat words like item, items, product, products, line item, description, contains, includes, "
                        "and bought as explicit item intent, and then filter via receipt_items.description.\n"
                        "If intent is still ambiguous and the user did not specify category/item, prefer matching category "
                        "name first; if that likely yields no result, use item-description matching as a fallback."
                    ),
                ),
                (
                    "human",
                    (
                        "Schema context:\n{schema_context}\n\n"
                        "Question:\n{question}\n\n"
                        "Return only the SQL query."
                    ),
                ),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def run(self, question: str, context: str) -> str:
        """
        Public API: build a read-only SQL query from user-friendly question + schema context.
        """
        raw_sql = self._build_sql(question=question, context=context)
        sql = self._sanitize_model_output(raw_sql)
        self._validate_read_only_sql(sql)
        return sql

    def _build_sql(self, question: str, context: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if not context or not context.strip():
            raise ValueError("Schema context cannot be empty.")

        return self._chain.invoke(
            {"question": question.strip(), "schema_context": context.strip()}
        )

    @staticmethod
    def _sanitize_model_output(raw_sql: str) -> str:
        sql = raw_sql.strip()
        if sql.startswith("```"):
            sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\s*```$", "", sql)
        return sql.strip().rstrip(";")

    @staticmethod
    def _validate_read_only_sql(sql: str) -> None:
        if not sql:
            raise ValueError("Generated SQL is empty.")

        normalized = sql.strip().lower()
        if not normalized.startswith(("select", "with")):
            raise ValueError("Generated SQL must start with SELECT or WITH.")

        if ";" in normalized:
            raise ValueError("Generated SQL must be a single statement.")

        forbidden_pattern = r"\b(" + "|".join(READ_ONLY_VERBS) + r")\b"
        if re.search(forbidden_pattern, normalized, flags=re.IGNORECASE):
            raise ValueError("Generated SQL contains non-read-only operation.")
