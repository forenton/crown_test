from __future__ import annotations

from schemas import ContractSummary, SummaryItem, SummarySource
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


def _deduplicate_items(items: object) -> list[SummaryItem]:
    items_by_value: dict[str, SummaryItem] = {}

    for item in items:
        if not isinstance(item, SummaryItem):
            item = SummaryItem.model_validate(item)

        normalized = _normalize_for_deduplication(item.value)
        if not normalized:
            continue

        if normalized not in items_by_value:
            items_by_value[normalized] = SummaryItem(
                value=item.value.strip(),
                sources=_deduplicate_sources(item.sources),
            )
            continue

        existing = items_by_value[normalized]
        existing.sources = _deduplicate_sources([*existing.sources, *item.sources])

    return list(items_by_value.values())


def _deduplicate_sources(sources: list[SummarySource]) -> list[SummarySource]:
    unique_sources = []
    seen = set()

    for source in sources:
        key = (source.clause, tuple(source.pages))
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)

    return unique_sources


def _normalize_for_deduplication(value: str) -> str:
    return " ".join(value.lower().split())
