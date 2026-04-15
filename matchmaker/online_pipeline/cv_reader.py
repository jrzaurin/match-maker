from pathlib import Path

import fitz


def read_pdf(path: str | Path) -> str:
    """Extract text from a PDF, preserving reading order across pages."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text("text", sort=True)
        pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)
