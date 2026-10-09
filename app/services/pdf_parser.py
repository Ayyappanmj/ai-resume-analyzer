"""PDF text extraction with pdfplumber."""
import io
import re

_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


def normalize_text(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text_from_pdf(data: bytes, max_pages: int = 10) -> tuple[str, int]:
    """Return (text, page_count). Raises on unreadable / encrypted PDFs."""
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages[:max_pages]:
            parts.append(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
    return normalize_text("\n".join(parts)), pages
