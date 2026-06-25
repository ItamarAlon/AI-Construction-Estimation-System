from langchain.agents.middleware import before_agent
from langchain_core.messages import HumanMessage
from pathlib import Path

import pymupdf as fitz
import base64
import re


def make_pdf_injection_middleware(max_edge: int | None = None):
    """Return a configured PDF injection middleware.

    max_edge: cap the longest rendered edge to this many pixels (zoom <= 4x).
              None keeps the original zoom=1 behaviour.
    """
    @before_agent
    def _middleware(state, runtime):
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

        if isinstance(content, str):
            pdf_match = re.search(r'["\']?[\w\\/: .()-]+\.pdf["\']?', content, re.IGNORECASE)
            if not pdf_match:
                return None
            pdf_blocks = _pdf_to_content_blocks(pdf_match.group(), max_edge)
            new_content = [{"type": "text", "text": content}] + pdf_blocks
            new_messages = messages[:-1] + [HumanMessage(content=new_content)]
            return {"messages": new_messages}

        if isinstance(content, list):
            # Skip only when PDF page blocks are already present — detected by the
            # "PDF: <name> (N page(s))" marker that _pdf_to_content_blocks emits.
            # We do NOT skip just because image blocks are present: those may be
            # segment crops, not PDF pages.
            if any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and re.search(r"^PDF: .+ \(\d+ page\(s\)\)$", b.get("text", ""))
                for b in content
            ):
                return None
            for i, block in enumerate(content):
                if block.get("type") != "text":
                    continue
                pdf_match = re.search(r'["\']?[\w\\/: .()-]+\.pdf["\']?',
                                       block.get("text", ""), re.IGNORECASE)
                if pdf_match:
                    pdf_blocks = _pdf_to_content_blocks(pdf_match.group(), max_edge)
                    new_content = content[:i + 1] + pdf_blocks + content[i + 1:]
                    new_messages = messages[:-1] + [HumanMessage(content=new_content)]
                    return {"messages": new_messages}

        return None

    return _middleware


# Default instance (zoom=1, no max-edge cap) — used by AgentBuilder.pdf_reader()
# when no max_edge is specified and by any direct import of the old name.
pdf_injection_middleware = make_pdf_injection_middleware()


def _pdf_to_content_blocks(pdf_path: str, max_edge: int | None = None) -> list:
    """Render every page of a PDF as image content blocks."""
    path = Path(pdf_path.strip("'\""))
    if not path.exists():
        return [{"type": "text", "text": f"File not found: {pdf_path}"}]
    doc = fitz.open(str(path))
    blocks = [{"type": "text", "text": f"PDF: {path.name} ({len(doc)} page(s))"}]
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            blocks.append({"type": "text", "text": f"Page {i} extracted text:\n{text}"})
        if max_edge is not None:
            longest_pt = max(page.rect.width, page.rect.height)
            zoom = min(4.0, max_edge / longest_pt) if longest_pt else 4.0
        else:
            zoom = 1
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        blocks.append({"type": "text", "text": f"Page {i} image:"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    doc.close()
    return blocks
