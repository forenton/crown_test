from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_PDF_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


class PdfValidationError(Exception):
    def __init__(self, detail: str, status_code: int = 400) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def validate_pdf_file(
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> dict[str, object]:
    if not filename or not filename.lower().endswith(".pdf"):
        raise PdfValidationError("File must have a .pdf extension")

    if content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise PdfValidationError("File content type must be application/pdf")

    if not content:
        raise PdfValidationError("File is empty")

    if len(content) > MAX_PDF_SIZE_BYTES:
        raise PdfValidationError("File exceeds 25 MB limit", status_code=413)

    if not content.startswith(b"%PDF-"):
        raise PdfValidationError("File does not have a valid PDF header")

    try:
        reader = PdfReader(BytesIO(content))
        page_count = len(reader.pages)
    except (PdfReadError, ValueError, TypeError, OSError) as exc:
        raise PdfValidationError("File is not a valid PDF") from exc

    if page_count < 1:
        raise PdfValidationError("PDF must contain at least one page")

    return {
        "valid": True,
        "filename": filename,
        "content_type": content_type,
        "size_kb": round(len(content) / 1024, 2),
        "page_count": page_count,
        "encrypted": bool(reader.is_encrypted),
    }
