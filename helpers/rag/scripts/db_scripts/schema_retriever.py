from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_embed_schema_module() -> ModuleType:
    source_path = Path(__file__).resolve().parents[2] / "db_setup" / "embed_schema.py"
    spec = importlib.util.spec_from_file_location("embed_schema_local", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {source_path}")

    module = importlib.util.module_from_spec(spec)
    # Ensure decorators/introspection can resolve the module during execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


get_schema_retriever = _load_embed_schema_module().get_schema_retriever


def retrieve_schema_for_question(question: str, k: int = 8) -> str:
    """
    Receives a natural-language question and returns the most relevant schema context.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    retriever = get_schema_retriever(k=k)
    return retriever.retrieve_context(question=question, k=k)

