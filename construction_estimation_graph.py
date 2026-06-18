from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from pdf_calculator_agent import agent as estimation_agent
from calculate_prices import extract_quantities, price_quantities, format_report


class State(TypedDict):
    pdf_path: str
    agent_output: str       # raw agent text (detection + classification + quantities JSON)
    quantities: dict        # parsed task name -> quantity
    breakdown: dict         # priced line items + grand total
    result: str             # human-readable cost report


def run_estimation(state: State) -> dict:
    # One vision agent: detects tasks AND measures quantities (no pricing).
    return {"agent_output": estimation_agent.run(state["pdf_path"])}


def run_pricing(state: State) -> dict:
    # Plain Python: turn quantities into costs. No LLM, no PDF.
    quantities = extract_quantities(state["agent_output"])
    breakdown = price_quantities(quantities)
    return {
        "quantities": quantities,
        "breakdown": breakdown,
        "result": format_report(breakdown),
    }


graph = (
    StateGraph(State)
    .add_node("estimation", run_estimation)
    .add_node("pricing", run_pricing)
    .add_edge(START, "estimation")
    .add_edge("estimation", "pricing")
    .add_edge("pricing", END)
    .compile()
)

#PDF_PATH = r"C:\Users\Alon\source\repos\Agentic_AI_2026\final_project\files\תכנית- פירוק הריסה ובנייה (1).pdf"
PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\הריסה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\בנייה (1).pdf"
#PDF_PATH = r"C:\Users\Alon\source\repos\Construction Estimation System\example_construction_pdfs\סט תוכניות (1).pdf"

if __name__ == "__main__":
    result = graph.invoke({"pdf_path": PDF_PATH})
    print("Result:", result["result"])
