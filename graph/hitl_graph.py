from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Literal, Annotated
import operator

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "agent_wrap"))

from agent_wrap.AgentBuilder import AgentBuilder
from agent_wrap.Agent import Agent
from get_key import get_openrouter_api_key


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    topic: str
    pending_question: str   # question the agent is waiting on; empty when done
    user_answer: str        # last answer provided by the human
    output: str
    log: Annotated[list[str], operator.add]


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def generate_quiz(topic: str, difficulty: str, num_questions: int) -> str:
    """Generate an exam quiz for a given topic, difficulty, and number of questions."""
    questions = [
        f"Q{i+1}: [{difficulty.capitalize()}] question about '{topic}'."
        for i in range(num_questions)
    ]
    return "\n".join(questions)


# ── Agent ──────────────────────────────────────────────────────────────────────

_model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
    temperature=0.2,
)

quiz_agent = (
    AgentBuilder(
        model=_model,
        tools=[generate_quiz],
        system_prompt=(
            "You are an exam quiz generator. "
            "When asked to generate a quiz, call the generate_quiz tool. "
            "If the user hasn't specified difficulty or number of questions, "
            "use ask_user to request the missing details before proceeding."
        ),
    )
    .ask_user_when_needed()
    .build()
)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def run_agent(state: State) -> dict:
    # On the first call use the topic; on subsequent calls feed the user's answer
    # back into the agent so it can resume from where it left off.
    text = state["user_answer"] if quiz_agent.is_interrupted() else state["topic"]
    result = quiz_agent.run(text)

    if quiz_agent.is_interrupted():
        return {
            "pending_question": result,
            "user_answer": "",
            "log": [f"Agent asked: {result}"],
        }
    return {
        "output": result,
        "pending_question": "",
        "log": ["Quiz generated"],
    }


def ask_human(state: State) -> dict:
    # Pause the graph and surface the agent's question to the caller.
    # Execution resumes when the caller provides Command(resume=<answer>).
    user_answer = interrupt(state["pending_question"])
    return {"user_answer": user_answer, "log": [f"User answered: {user_answer}"]}


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_agent(state: State) -> Literal["ask_human", "__end__"]:
    return "ask_human" if state["pending_question"] else END


# ── Graph ──────────────────────────────────────────────────────────────────────

graph = (
    StateGraph(State)
    .add_node("run_agent", run_agent)
    .add_node("ask_human", ask_human)
    .add_edge(START, "run_agent")
    .add_conditional_edges("run_agent", route_after_agent, ["ask_human", END])
    .add_edge("ask_human", "run_agent")
    .compile(checkpointer=MemorySaver())
)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "topic": "Generate a quiz for me on the French Revolution",
        "pending_question": "",
        "user_answer": "",
        "output": "",
        "log": [],
    }

    result = graph.invoke(initial_state, config)

    # Drive the HITL loop: keep answering until the graph finishes.
    while result.get("__interrupt__"):
        question = result["__interrupt__"][0].value
        print(f"\nAgent: {question}")
        answer = input("You: ").strip()
        result = graph.invoke(Command(resume=answer), config)

    print("\n--- Quiz ---")
    print(result["output"])
    print("\n--- Log ---")
    for entry in result["log"]:
        print(" •", entry)
