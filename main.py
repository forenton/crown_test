from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from schemas import ContractSummaryResponse
from services.contract_summarizer import summarize_contract_fragments_by_category
from services.ollama_client import OllamaError
from services.pdf_extractor import PdfTextExtractionError, build_relevant_fragments, extract_pdf_pages
from services.pdf_validator import PdfValidationError, validate_pdf_file
from services.settings import (
    get_ollama_max_fragment_chars,
    get_ollama_max_fragments_per_category,
    get_ollama_model,
)

app = FastAPI(
    title="Crown Contract Summarizer",
    version="0.1.0",
    description="API for summarizing contracts in PDF.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize_contract", response_model=ContractSummaryResponse, tags=["pdf"])
async def summarize_contract(
    file: Annotated[UploadFile, File(description="PDF file to summarize")],
) -> dict[str, object]:
    content = await file.read()

    try:
        validation = validate_pdf_file(file.filename, file.content_type, content)
        extraction = extract_pdf_pages(content)
        fragments_by_category = build_relevant_fragments(
            extraction["pages"],
            max_fragments_per_category=get_ollama_max_fragments_per_category(),
            max_fragment_chars=get_ollama_max_fragment_chars(),
        )
        summary = await summarize_contract_fragments_by_category(fragments_by_category)
    except (PdfValidationError, PdfTextExtractionError, OllamaError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    processed_fragments_by_category = {
        category: len(fragments)
        for category, fragments in fragments_by_category.items()
    }

    return {
        **validation,
        "page_count": extraction["page_count"],
        "extracted_pages": len(extraction["pages"]),
        "processed_fragments": sum(processed_fragments_by_category.values()),
        "processed_fragments_by_category": processed_fragments_by_category,
        "model": get_ollama_model(),
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
