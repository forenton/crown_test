from __future__ import annotations

from schemas import ContractSummary
from services.ollama_client import summarize_contract_fragments
from services.pdf_extractor import PdfTextFragment


async def summarize_contract_fragments_by_category(
    fragments_by_category: dict[str, list[PdfTextFragment]],
) -> ContractSummary:
    summary = await summarize_contract_fragments(fragments_by_category)
    return ContractSummary(
        contract_amount=summary.contract_amount,
        performance_terms=_deduplicate_items(summary.performance_terms),
        contractor_requirements=_deduplicate_items(summary.contractor_requirements),
        penalties=_deduplicate_items(summary.penalties),
    )


def _deduplicate_items(items: object) -> list[str]:
    unique_items = []
    seen = set()

    for item in items:
        normalized = _normalize_for_deduplication(str(item))
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique_items.append(str(item).strip())

    return unique_items


def _normalize_for_deduplication(value: str) -> str:
    return " ".join(value.lower().split())
