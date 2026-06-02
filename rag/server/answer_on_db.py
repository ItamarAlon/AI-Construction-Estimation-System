from __future__ import annotations

from agents.query_result_answer_agent import QueryResultAnswerAgent
from agents.sql_query_agent import ReadOnlySQLAgent
from scripts.db_scripts.db_query_runner import run_read_query
from scripts.db_scripts.schema_retriever import retrieve_schema_for_question


def answer_on_db(question: str) -> str:
    question = question.lower()
    schema_context = retrieve_schema_for_question(question)
    sql_agent = ReadOnlySQLAgent()
    answer_agent = QueryResultAnswerAgent()

    sql = sql_agent.run(question, schema_context)
    query_result = run_read_query(sql)
    print(f"query_result: {query_result}")
    answer = answer_agent.run(question, query_result)
    return answer
