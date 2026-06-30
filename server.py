import tempfile
import shutil
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from construction_tasks_prices.read_construction_tasks_prices import (
    add_task,
    remove_task,
    update_task_price,
    toggle_task_type,
    get_construction_tasks_prices,
)
from graph.construction_estimation_graph import graph

app = FastAPI(title="Construction Estimation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EstimateRequest(BaseModel):
    pdf_path: str
    pages: list[int] | None = None   # 1-indexed pages to analyze; None/empty = all
    show_measurements: bool = False
    scale_factor: float = 1.0        # multiplier for per-meter measurements (1.0 = no correction)

class AnnotatedPage(BaseModel):
    page: int
    base_image_b64: str
    task_layers: dict[str, str] = {}   # task_name -> png_b64 (white bg, multiply in UI)

class LegendEntry(BaseModel):
    task: str
    color: str

class LineItem(BaseModel):
    task: str
    quantity: float
    unit_price: float | None = None
    cost: float | None = None

class PageBreakdown(BaseModel):
    page: int
    line_items: list[LineItem]
    subtotal: float

class EstimateResponse(BaseModel):
    agent_output: str
    result: str
    line_items: list[LineItem] = []
    grand_total: float = 0.0
    page_breakdowns: list[PageBreakdown] = []
    annotated_pages: list[AnnotatedPage] = []
    legend: list[LegendEntry] = []
    scale_factor: float = 1.0          # auto-detected or user-provided scale correction

class AddTaskRequest(BaseModel):
    name: str
    price: float
    per_meter: bool

class UpdateTaskRequest(BaseModel):
    price: float


@app.get("/tasks")
def get_tasks():
    return get_construction_tasks_prices()


@app.post("/tasks", status_code=201)
def add_new_task(request: AddTaskRequest):
    try:
        add_task(request.name, request.price, request.per_meter)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"message": "Task added successfully."}


@app.put("/tasks/{task_name}")
def update_task(task_name: str, request: UpdateTaskRequest):
    try:
        update_task_price(task_name, request.price)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Task '{task_name}' updated successfully."}


@app.patch("/tasks/{task_name}/toggle-type")
def toggle_task_type_endpoint(task_name: str):
    try:
        toggle_task_type(task_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Task '{task_name}' type toggled."}


@app.delete("/tasks/{task_name}")
def delete_task(task_name: str):
    try:
        remove_task(task_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Task '{task_name}' removed successfully."}


def _make_line_items(raw_items: list) -> list[LineItem]:
    return [
        LineItem(
            task=item["task"],
            quantity=item["quantity"],
            unit_price=item.get("unit_price"),
            cost=item.get("cost"),
        )
        for item in raw_items
        if "error" not in item
    ]


def _to_response(state: dict) -> EstimateResponse:
    annotations = state.get("annotations") or {}
    breakdown = state.get("calculated_prices_breakdown") or {}
    line_items = _make_line_items(breakdown.get("line_items", []))
    page_breakdowns = [
        PageBreakdown(
            page=page,
            line_items=_make_line_items(page_bd.get("line_items", [])),
            subtotal=page_bd.get("grand_total", 0.0),
        )
        for page, page_bd in sorted(breakdown.get("pages", {}).items())
    ]
    return EstimateResponse(
        agent_output=state.get("agent_output", ""),
        result=state["result"],
        line_items=line_items,
        grand_total=breakdown.get("grand_total", 0.0),
        page_breakdowns=page_breakdowns,
        annotated_pages=annotations.get("pages", []),
        legend=annotations.get("legend", []),
        scale_factor=state.get("scale_factor", 1.0) or 1.0,
    )


@app.post("/estimate", response_model=EstimateResponse)
def estimate(request: EstimateRequest):
    if not Path(request.pdf_path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {request.pdf_path}")
    state = graph.invoke({
        "pdf_path": request.pdf_path,
        "pages": request.pages or [],
        "show_measurements": request.show_measurements,
        "scale_factor": request.scale_factor,
    })
    return _to_response(state)


def _parse_pages(pages: str | None) -> list[int]:
    """Parse a comma-separated page string like '1,3,5' into [1,3,5] (empty = all)."""
    if not pages:
        return []
    out = []
    for part in pages.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


@app.post("/estimate/upload", response_model=EstimateResponse)
async def estimate_upload(
    file: UploadFile = File(...),
    pages: str | None = Form(None),
    show_measurements: bool = Form(False),
    scale_factor: float = Form(1.0),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        state = graph.invoke({
            "pdf_path": tmp.name,
            "pages": _parse_pages(pages),
            "show_measurements": show_measurements,
            "scale_factor": scale_factor,
        })
        return _to_response(state)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
