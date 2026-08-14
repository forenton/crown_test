from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ContractSummary(BaseModel):
    contract_amount: str | None = Field(
        default=None,
        description="Contract amount with currency, if found.",
    )
    performance_terms: list[str] = Field(
        default_factory=list,
        description="Deadlines and performance terms.",
    )
    contractor_requirements: list[str] = Field(
        default_factory=list,
        description="Key requirements for the contractor.",
    )
    penalties: list[str] = Field(
        default_factory=list,
        description="Fines, penalties, and liability conditions.",
    )

    @field_validator("contract_amount", mode="before")
    @classmethod
    def cast_contract_amount(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @field_validator("performance_terms", "contractor_requirements", "penalties", mode="before")
    @classmethod
    def cast_list_items(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]


class ContractSummaryResponse(BaseModel):
    valid: bool
    filename: str
    content_type: str | None
    size_bytes: int
    page_count: int
    extracted_pages: int
    processed_fragments: int
    processed_fragments_by_category: dict[str, int]
    encrypted: bool
    model: str
    summary: ContractSummary
