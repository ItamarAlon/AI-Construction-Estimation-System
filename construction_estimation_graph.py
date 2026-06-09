import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from task_finder_agent import agent as task_finder_agent
from pdf_calculator_agent import agent as pdf_calculator_agent


class State(TypedDict):
    pdf_path: str
    detected_tasks: str
    result: str


def parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


def run_task_finder(state: State) -> dict:
    return {"detected_tasks": task_finder_agent.run(state["pdf_path"])}


def run_pdf_calculator(state: State) -> dict:
    tasks_dict = parse_json(state['detected_tasks'])
    msg = f"{state['pdf_path']}\n\n'Building Wall (per meter)-'{tasks_dict["Building Wall (per meter)"]}"
    print(msg)
    return {"result": pdf_calculator_agent.run(msg)}


graph = (
    StateGraph(State)
    .add_node("task_finder", run_task_finder)
    .add_node("pdf_calculator", run_pdf_calculator)
    .add_edge(START, "task_finder")
    .add_edge("task_finder", "pdf_calculator")
    .add_edge("pdf_calculator", END)
    .compile()
)

PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"

if __name__ == "__main__":
    result = graph.invoke({"pdf_path": PDF_PATH})
    print("Result:", result["result"])
