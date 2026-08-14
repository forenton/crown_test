from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from schemas import ContractSummary
from services.pdf_extractor import PdfTextFragment
from services.settings import (
    SettingsError,
    get_ollama_base_url,
    get_ollama_model,
    get_ollama_num_predict,
    get_ollama_temperature,
    get_ollama_timeout_seconds,
)


SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "contract_amount": {
            "anyOf": [
                {"type": "null"},
                {"$ref": "#/$defs/summary_item"},
            ],
        },
        "performance_terms": {
            "type": "array",
            "items": {"$ref": "#/$defs/summary_item"},
        },
        "contractor_requirements": {
            "type": "array",
            "items": {"$ref": "#/$defs/summary_item"},
        },
        "penalties": {
            "type": "array",
            "items": {"$ref": "#/$defs/summary_item"},
        },
    },
    "required": [
        "contract_amount",
        "performance_terms",
        "contractor_requirements",
        "penalties",
    ],
    "additionalProperties": False,
    "$defs": {
        "summary_item": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clause": {
                                "anyOf": [
                                    {"type": "null"},
                                    {"type": "string"},
                                ],
                            },
                            "pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["clause", "pages"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["value", "sources"],
            "additionalProperties": False,
        },
    },
}


class OllamaError(Exception):
    def __init__(self, detail: str, status_code: int = 502) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


async def summarize_contract_fragments(
    fragments_by_category: dict[str, list[PdfTextFragment]],
    model: str | None = None,
) -> ContractSummary:
    prompt = _build_summary_prompt(fragments_by_category)
    return await _request_contract_summary(prompt=prompt, model=model)


async def _request_contract_summary(prompt: str, model: str | None = None) -> ContractSummary:
    try:
        selected_model = model or get_ollama_model()
        base_url = get_ollama_base_url()
        timeout_seconds = get_ollama_timeout_seconds()
        temperature = get_ollama_temperature()
        num_predict = get_ollama_num_predict()
    except SettingsError as exc:
        raise OllamaError(str(exc), 500) from exc

    payload = {
        "model": selected_model,
        "stream": False,
        "format": SUMMARY_JSON_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "Ты анализируешь российские государственные контракты. "
                    "Отвечай строго валидным JSON без markdown и без рассуждений. "
                    "Запрещено додумывать: используй только факты из переданных фрагментов."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaError("Could not connect to Ollama. Is Ollama running?", 503) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama request timed out after {timeout_seconds} seconds. "
            "Reduce OLLAMA_MAX_FRAGMENT_CHARS, OLLAMA_MAX_FRAGMENTS_PER_CATEGORY, "
            "or use a smaller local model.",
            504,
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ollama returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise OllamaError("Ollama request failed") from exc

    try:
        content = response.json()["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaError("Ollama returned an unexpected response") from exc

    return _parse_contract_summary(content)


def _build_summary_prompt(fragments_by_category: dict[str, list[PdfTextFragment]]) -> str:
    return (
        "/no_think\n"
        "Извлеки из фрагментов только явно указанные данные:\n"
        "- contract_amount: только цена/сумма контракта; не НМЦК, не обеспечение и не штраф;\n"
        "- performance_terms: сроки выполнения, поставки, оказания услуг, этапы, даты начала/окончания;\n"
        "- contractor_requirements: ключевые требования и обязанности исполнителя/поставщика/подрядчика;\n"
        "- penalties: штрафы, пени, неустойки и условия ответственности.\n"
        "Для каждого найденного значения обязательно укажи источники sources: номер пункта clause и страницы pages. "
        "Бери clause и pages только из заголовка фрагмента. Если clause в заголовке null, верни null.\n"
        "Если поле не найдено в переданных фрагментах, верни null для суммы или пустой список. "
        "Не используй типовые формулировки, которых нет во фрагментах.\n"
        "Верни JSON строго такой структуры:\n"
        "{\n"
        '  "contract_amount": {\n'
        '    "value": "сумма контракта строкой",\n'
        '    "sources": [{"clause": "номер пункта или null", "pages": [1]}]\n'
        "  },\n"
        '  "performance_terms": [\n'
        '    {"value": "срок выполнения", "sources": [{"clause": "1.4", "pages": [1]}]}\n'
        "  ],\n"
        '  "contractor_requirements": [\n'
        '    {"value": "требование", "sources": [{"clause": "2.1.1", "pages": [1]}]}\n'
        "  ],\n"
        '  "penalties": [\n'
        '    {"value": "штраф или пеня", "sources": [{"clause": "10.2", "pages": [10]}]}\n'
        "  ]\n"
        "}\n\n"
        f"Фрагменты договора:\n{_format_fragments_by_category(fragments_by_category)}"
    )


def _format_fragments_by_category(
    fragments_by_category: dict[str, list[PdfTextFragment]],
) -> str:
    sections = []
    for category in ("amount", "terms", "requirements", "penalties"):
        fragments = fragments_by_category.get(category, [])
        sections.append(f"## {category}")
        if not fragments:
            sections.append("Фрагменты не найдены.")
            continue

        for fragment in fragments:
            pages = ", ".join(str(page) for page in fragment.page_numbers)
            sections.append(
                f"[fragment={fragment.fragment_id}; clause={fragment.clause or 'null'}; "
                f"pages=[{pages}]; score={fragment.score}]\n"
                f"{fragment.text}"
            )

    return "\n\n---\n\n".join(sections)


def _parse_contract_summary(content: str) -> ContractSummary:
    data = _json_loads_lenient(content)
    try:
        return ContractSummary.model_validate(data)
    except ValidationError as exc:
        raise OllamaError(f"Ollama returned invalid summary schema: {exc}") from exc


def _json_loads_lenient(content: str) -> dict[str, Any]:
    try:
        return _ensure_json_object(json.loads(content))
    except json.JSONDecodeError as original_exc:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise OllamaError("Ollama did not return JSON") from original_exc
        try:
            return _ensure_json_object(json.loads(match.group(0)))
        except json.JSONDecodeError as extracted_exc:
            raise OllamaError(
                "Ollama returned malformed JSON. Increase OLLAMA_NUM_PREDICT "
                "or reduce OLLAMA_MAX_FRAGMENTS_PER_CATEGORY / OLLAMA_MAX_FRAGMENT_CHARS."
            ) from extracted_exc


def _ensure_json_object(loaded: Any) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        raise OllamaError("Ollama JSON response must be an object")

    return loaded
