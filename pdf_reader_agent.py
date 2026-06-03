from pathlib import Path
import sys
import base64
import re

# File is at repo root; add helpers/ and helpers/agent_wrap/ so package and
# bare imports inside AgentBuilder resolve correctly.
_root = Path(__file__).resolve().parent
for _p in [str(_root / "helpers"), str(_root / "helpers" / "agent_wrap")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_openai import ChatOpenAI
from helpers.agent_wrap.AgentBuilder import AgentBuilder
from get_key import get_openrouter_api_key
import pymupdf as fitz


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
            #print("written pdf text: ", text)
            blocks.append({"type": "text", "text": f"Page {i} extracted text:\n{text}"})
        # Render at 4× resolution — needed for small Hebrew text and fine plan details
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        blocks.append({"type": "text", "text": f"Page {i} image:"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    doc.close()
    return blocks


def _build_input(user_text: str) -> dict:
    """If the message contains a PDF path, embed the page images in the user message."""
    pdf_match = re.search(r'["\']?[\w\\/: .()-]+\.pdf["\']?', user_text, re.IGNORECASE)

    if pdf_match:
        pdf_blocks = _pdf_to_content_blocks(pdf_match.group())
        content = [{"type": "text", "text": user_text}] + pdf_blocks
    else:
        content = user_text

    return {"messages": [{"role": "user", "content": content}]}


model = ChatOpenAI(
    model="openai/gpt-4o",   # vision-capable model needed to analyse the page images
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=get_openrouter_api_key(),
)

agent = AgentBuilder(
    model=model,
    system_prompt=(
        "You are a helpful assistant that reads and analyses construction plan PDFs. "
        "The user may include PDF page images in their message. Analyse them thoroughly: "
        "identify room names, dimensions, structural elements, annotations, and spatial layout."
        "The user might ask questions about the content of the PDF. Analys the PDF page images first and then answer based on your analysis."
    ),
).with_memory().build()


if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        raw = agent.invoke(_build_input(user_input), config=agent._config, version="v2")
        print("Agent:", agent._process_result(raw))
