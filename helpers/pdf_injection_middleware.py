from langchain.agents.middleware import before_agent
from langchain_core.messages import HumanMessage
from pathlib import Path

import pymupdf as fitz
import base64
import re

@before_agent
def pdf_injection_middleware(state, runtime):
    """Detect a PDF path in the last user message and expand it into content blocks."""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    if isinstance(last_msg, HumanMessage):
        content = last_msg.content
    elif isinstance(last_msg, dict) and last_msg.get("role") == "user":
        content = last_msg.get("content", "")
    else:
        return None

    if not isinstance(content, str):
        return None

    pdf_match = re.search(r'["\']?[\w\\/: .()-]+\.pdf["\']?', content, re.IGNORECASE)
    if not pdf_match:
        return None

    pdf_blocks = _pdf_to_content_blocks(pdf_match.group())
    new_content = [{"type": "text", "text": content}] + pdf_blocks
    new_messages = messages[:-1] + [HumanMessage(content=new_content)]
    return {"messages": new_messages}

def _pdf_to_content_blocks(pdf_path: str) -> list:
    """Render every page of a PDF as an image and return content blocks."""
    path = Path(pdf_path.strip("'\""))
    if not path.exists():
        return [{"type": "text", "text": f"File not found: {pdf_path}"}]
    doc = fitz.open(str(path))
    blocks = [{"type": "text", "text": f"PDF: {path.name} ({len(doc)} page(s))"}]
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            blocks.append({"type": "text", "text": f"Page {i} extracted text:\n{text}"})
        # Render at 4× resolution for small Hebrew text and fine plan details
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        blocks.append({"type": "text", "text": f"Page {i} image:"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    doc.close()
    return blocks