from __future__ import annotations

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    question_type: str          # "factual" | "conceptual"
    answer: str
    reasoning_steps: Annotated[list[str], operator.add]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_question(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(
            "You are a question classifier for an exam prep system. "
            "Reply with exactly one word: 'factual' if the question asks for a specific fact, "
            "date, name, or definition; 'conceptual' if it asks for explanation, comparison, or reasoning."
        ),
        HumanMessage(state["question"]),
    ])
    question_type = response.content.strip().lower()
    if question_type not in ("factual", "conceptual"):
        question_type = "conceptual"
    return {
        "question_type": question_type,
        "reasoning_steps": [f"Classified as: {question_type}"],
    }


def route_by_type(state: State) -> Literal["factual_answer", "conceptual_answer"]:
    return f"{state['question_type']}_answer"  # type: ignore[return-value]


def factual_answer(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(
            "You are a precise exam prep assistant. "
            "Give a concise, accurate answer to the factual question. "
            "Keep it under 3 sentences."
        ),
        HumanMessage(state["question"]),
    ])
    return {
        "answer": response.content,
        "reasoning_steps": ["Answered with factual strategy"],
    }


def conceptual_answer(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(
            "You are a thorough exam prep tutor. "
            "Explain the concept clearly with structure: define, explain why it matters, give an example. "
            "Use bullet points where helpful."
        ),
        HumanMessage(state["question"]),
    ])
    return {
        "answer": response.content,
        "reasoning_steps": ["Answered with conceptual strategy"],
    }


graph = (
    StateGraph(State)
    .add_node("classify", classify_question)
    .add_node("factual_answer", factual_answer)
    .add_node("conceptual_answer", conceptual_answer)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_by_type, ["factual_answer", "conceptual_answer"])
    .add_edge("factual_answer", END)
    .add_edge("conceptual_answer", END)
    .compile()
)


if __name__ == "__main__":
    questions = [
        "In what year did World War II end?",
        "Why does increasing interest rates reduce inflation?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = graph.invoke({"question": q, "reasoning_steps": []})
        print(f"Type:     {result['question_type']}")
        print(f"Steps:    {result['reasoning_steps']}")
        print(f"Answer:\n{result['answer']}")
        print("-" * 60)
