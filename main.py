from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from services.pdf_validator  import PdfValidationError, validate_pdf_file

app = FastAPI(
    title="Crown PDF Validator",
    version="0.1.0",
    description="API for summarizing contracts in PDF.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize_contract", tags=["pdf"])
async def summarize_contract(
    file: Annotated[UploadFile, File(description="PDF file to summarize")],
) -> dict[str, object]:
    content = await file.read()

    try:
        return validate_pdf_file(file.filename, file.content_type, content)
    except PdfValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
