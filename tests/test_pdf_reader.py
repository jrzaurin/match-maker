from pathlib import Path

import fitz

from matchmaker.online_pipeline.cv_reader import read_pdf


def test_read_pdf_extracts_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cv.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello CV")
    doc.save(pdf_path)
    doc.close()

    text = read_pdf(pdf_path)
    assert "Hello CV" in text
