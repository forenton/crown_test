from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SummarySource(BaseModel):
    clause: str | None = Field(
        default=None,
        description="Contract clause number, if available.",
    )
    pages: list[int] = Field(
        default_factory=list,
        description="Source contract pages.",
    )


class SummaryItem(BaseModel):
    value: str = Field(description="Extracted contract fact.")
    sources: list[SummarySource] = Field(
        default_factory=list,
        description="Source clauses and pages.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_sources(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "sources" not in value and "source" in value:
            value = dict(value)
            value["sources"] = [value.pop("source")]
        return value

    @field_validator("value", mode="before")
    @classmethod
    def cast_value(cls, value: Any) -> str:
        return str(value)

    @field_validator("sources", mode="before")
    @classmethod
    def cast_sources(cls, value: Any) -> list[SummarySource]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        sources = []
        for source in value:
            if source is None:
                continue
            if isinstance(source, SummarySource):
                sources.append(source)
            else:
                sources.append(SummarySource.model_validate(source))
        return sources


class ContractSummary(BaseModel):
    contract_amount: SummaryItem | None = Field(
        default=None,
        description="Contract amount with source, if found.",
    )
    performance_terms: list[SummaryItem] = Field(
        default_factory=list,
        description="Deadlines and performance terms with sources.",
    )
    contractor_requirements: list[SummaryItem] = Field(
        default_factory=list,
        description="Key contractor requirements with sources.",
    )
    penalties: list[SummaryItem] = Field(
        default_factory=list,
        description="Fines, penalties, and liability conditions with sources.",
    )

    @field_validator("contract_amount", mode="before")
    @classmethod
    def cast_contract_amount(cls, value: Any) -> SummaryItem | None:
        if value is None:
            return None
        if isinstance(value, SummaryItem):
            return value
        if isinstance(value, dict):
            return SummaryItem.model_validate(value)
        return SummaryItem(value=str(value), sources=[])

    @field_validator("performance_terms", "contractor_requirements", "penalties", mode="before")
    @classmethod
    def cast_list_items(cls, value: Any) -> list[SummaryItem]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, SummaryItem):
                items.append(item)
                continue
            if isinstance(item, dict):
                items.append(SummaryItem.model_validate(item))
            else:
                items.append(SummaryItem(value=str(item), sources=[]))
        return items


class ContractSummaryResponse(BaseModel):
    valid: bool
    filename: str
    content_type: str | None
    size_kb: float
    page_count: int
    extracted_pages: int
    processed_fragments: int
    processed_fragments_by_category: dict[str, int]
    encrypted: bool
    model: str
    summary: ContractSummary
