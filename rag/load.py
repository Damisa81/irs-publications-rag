"""Step 1: load IRS PDFs into one record per page, and strip printer boilerplate."""
import re
from pathlib import Path

from pypdf import PdfReader

# Printer-proof boilerplate on every IRS page (and a variant on the cover page): not
# content, and left in it would pollute every chunk's embedding.
PROOF_LINES = re.compile(
    r"^(.*\bFileid:.*"
    r"|The type and rule above prints on all proofs.*"
    r"|.*MUST be removed before printing\.?)$",
    re.MULTILINE,
)


def load_pages(pdf_path: Path) -> list[dict]:
    """One record per page, keeping where the text came from so answers can cite it."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():                                  # skip blank pages
            pages.append({"doc_id": pdf_path.stem, "page": i, "text": text})
    return pages


def clean(text: str) -> str:
    text = PROOF_LINES.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse runs of blank lines
    return text.strip()
