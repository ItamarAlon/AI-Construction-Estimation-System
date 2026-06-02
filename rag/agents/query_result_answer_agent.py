from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class QueryResultAnswerAgent:
    """
    Converts SQL query results into a natural-language answer.
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
                        "You are a data assistant that answers user questions based ONLY on SQL query results.\n"
                        "Use the provided question and rows to write a clear, concise natural-language answer.\n"
                        "Do not invent values that are not present in the rows.\n"
                        "If rows are empty, say that no matching records were found.\n"
                        "If rows are NOT empty, do not claim that no records were found.\n"
                        "For yes/no style questions, answer 'Yes' when rows are non-empty and 'No' when rows are empty"
                    ),
                ),
                (
                    "human",
                    (
                        "Original question:\n{question}\n\n"
                        "SQL query result rows (JSON):\n{query_result_json}\n\n"
                        "Write the final answer for the user."
                    ),
                ),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def run(self, question: str, query_result: list[dict[str, Any]]) -> str:
        """
        Public API: produce a natural-language answer from question + SQL result rows.
        """
        if self._is_yes_no_question(question):
            return self._build_yes_no_answer(query_result)

        query_result_json = self._prepare_inputs(question=question, query_result=query_result)
        answer = self._chain.invoke(
            {"question": question.strip(), "query_result_json": query_result_json}
        )
        return answer.strip()

    @staticmethod
    def _prepare_inputs(question: str, query_result: list[dict[str, Any]]) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if not isinstance(query_result, list):
            raise ValueError("Query result must be a list of row dictionaries.")

        return json.dumps(query_result, ensure_ascii=True, default=str)

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        normalized = question.strip().lower()
        return bool(
            re.match(
                r"^(was|were|is|are|did|does|do|has|have|had|can|could|should|would)\b",
                normalized,
            )
        )

    @staticmethod
    def _build_yes_no_answer(query_result: list[dict[str, Any]]) -> str:
        if not query_result:
            return "No, no matching records were found."

        first_row = query_result[0] if isinstance(query_result[0], dict) else {}
        evidence_parts: list[str] = []
        for key in ("merchant_name", "name", "description", "receipt_date", "total_amount"):
            value = first_row.get(key)
            if value is not None and str(value).strip():
                label = key.replace("_", " ")
                evidence_parts.append(f"{label}: {value}")

        evidence = "; ".join(evidence_parts) if evidence_parts else "at least one matching row exists"
        return f"Yes, matching records were found ({len(query_result)}). Example: {evidence}."
