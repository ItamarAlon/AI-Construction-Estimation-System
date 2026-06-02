from __future__ import annotations

from typing import Annotated
import operator

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from agent_wrap.Agent import Agent
from get_key import get_openrouter_api_key


class State(TypedDict):
    topic: str
    notes: str
    questions: str
    answers: str
    log: Annotated[list[str], operator.add]


_model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
    temperature=0.3,
)

notes_agent = Agent(
    model=_model,
    system_prompt=(
        "You are a study notes writer. "
        "Given a topic, produce clear, structured bullet-point study notes "
        "covering the key concepts, definitions, and important facts. "
        "Keep it concise — max 300 words."
    ),
)

question_agent = Agent(
    model=_model,
    system_prompt=(
        "You are an exam question writer. "
        "Given study notes, generate 5 exam-style questions (mix of factual and conceptual). "
        "Number them 1–5. Do not include answers."
    ),
)

answer_agent = Agent(
    model=_model,
    system_prompt=(
        "You are an exam tutor. "
        "Given a numbered list of exam questions, provide a clear, accurate answer for each one. "
        "Label each answer with its matching number."
    ),
)


def generate_notes(state: State) -> dict:
    notes = notes_agent.srun(f"Topic: {state['topic']}")
    return {"notes": notes, "log": ["Notes generated"]}


def generate_questions(state: State) -> dict:
    questions = question_agent.srun(f"Study notes:\n{state['notes']}")
    return {"questions": questions, "log": ["Questions generated"]}


def generate_answers(state: State) -> dict:
    answers = answer_agent.srun(f"Questions:\n{state['questions']}")
    return {"answers": answers, "log": ["Answers generated"]}


graph = (
    StateGraph(State)
    .add_node("notes", generate_notes)
    .add_node("questions", generate_questions)
    .add_node("answers", generate_answers)
    .add_edge(START, "notes")
    .add_edge("notes", "questions")
    .add_edge("questions", "answers")
    .add_edge("answers", END)
    .compile()
)


if __name__ == "__main__":
    topic = "The French Revolution"
    print(f"Topic: {topic}\n{'=' * 60}")

    result = graph.invoke({"topic": topic, "log": []})

    print("STUDY NOTES:\n", result["notes"])
    print("\nEXAM QUESTIONS:\n", result["questions"])
    print("\nANSWERS:\n", result["answers"])
    print("\nPipeline log:", result["log"])
