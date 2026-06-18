from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from pdf_calculator_agent import agent as estimation_agent


class State(TypedDict):
    pdf_path: str
    result: str


def run_estimation(state: State) -> dict:
    # One agent now detects tasks AND costs them from a single PDF read.
    return {"result": estimation_agent.run(state["pdf_path"])}


graph = (
    StateGraph(State)
    .add_node("estimation", run_estimation)
    .add_edge(START, "estimation")
    .add_edge("estimation", END)
    .compile()
)

#PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\הריסה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\בנייה (1).pdf"
PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\סט תוכניות (1).pdf"

if __name__ == "__main__":
    result = graph.invoke({"pdf_path": PDF_PATH})
    print("Result:", result["result"])
