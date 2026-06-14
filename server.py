import tempfile
import shutil
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from construction_tasks_prices.read_construction_tasks_prices import (
    add_task,
    remove_task,
    update_task_price,
    get_construction_tasks_prices,
)
from construction_estimation_graph import graph

app = FastAPI(title="Construction Estimation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EstimateRequest(BaseModel):
    pdf_path: str

class EstimateResponse(BaseModel):
    detected_tasks: str
    result: str

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


@app.delete("/tasks/{task_name}")
def delete_task(task_name: str):
    try:
        remove_task(task_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Task '{task_name}' removed successfully."}


@app.post("/estimate", response_model=EstimateResponse)
def estimate(request: EstimateRequest):
    if not Path(request.pdf_path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {request.pdf_path}")
    state = graph.invoke({"pdf_path": request.pdf_path})
    return EstimateResponse(detected_tasks=state["detected_tasks"], result=state["result"])


@app.post("/estimate/upload", response_model=EstimateResponse)
async def estimate_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        state = graph.invoke({"pdf_path": tmp.name})
        return EstimateResponse(
            detected_tasks=state["detected_tasks"],
            result=state["result"],
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)
