from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from schemas import ContractSummary, SummaryItem
from services.pdf_extractor import PdfTextFragment
from services.settings import (
    SettingsError,
    get_ollama_base_url,
    get_ollama_model,
    get_ollama_num_predict,
    get_ollama_temperature,
    get_ollama_timeout_seconds,
)


logger = logging.getLogger(__name__)


MODEL_ITEM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "fragment_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["value", "fragment_ids"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    response_field: str
    description: str
    empty_value: str
    max_results: int
    num_predict: int


CATEGORY_CONFIGS = (
    CategoryConfig(
        name="amount",
        response_field="contract_amount",
        description=(
            "Найди только итоговую цену/сумму контракта. Игнорируй НМЦК, "
            "обеспечение, гарантии, штрафы, пени и посторонние суммы."
        ),
        empty_value="null",
        max_results=1,
        num_predict=500,
    ),
    CategoryConfig(
        name="terms",
        response_field="performance_terms",
        description=(
            "Найди основной срок исполнения контракта: срок выполнения работ, поставки товара "
            "или оказания услуг. В первую очередь выбирай сроки из пунктов о предмете, сроках "
            "поставки/выполнения или графике выполнения обязательств. Не включай второстепенные "
            "сроки приемки, оплаты, устранения недостатков, претензий, уведомлений, действия "
            "контракта, форс-мажора и гарантий, если найден основной срок исполнения."
        ),
        empty_value="[]",
        max_results=3,
        num_predict=1000,
    ),
    CategoryConfig(
        name="requirements",
        response_field="contractor_requirements",
        description=(
            "Найди ключевые требования и обязанности исполнителя, поставщика или подрядчика."
        ),
        empty_value="[]",
        max_results=5,
        num_predict=1400,
    ),
    CategoryConfig(
        name="penalties",
        response_field="penalties",
        description="Найди штрафы, пени, неустойки и условия ответственности.",
        empty_value="[]",
        max_results=5,
        num_predict=1400,
    ),
)


class OllamaError(Exception):
    def __init__(self, detail: str, status_code: int = 502) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class OllamaRetryableError(OllamaError):
    pass


class OllamaResponseFormatError(OllamaRetryableError):
    pass


class OllamaTransientHTTPError(OllamaRetryableError):
    pass


async def summarize_contract_fragments(
    fragments_by_category: dict[str, list[PdfTextFragment]],
    model: str | None = None,
) -> ContractSummary:
    summary = ContractSummary()

    for config in CATEGORY_CONFIGS:
        fragments = fragments_by_category.get(config.name, [])
        last_error: OllamaRetryableError | None = None

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                result = await analyze_category(
                    config=config,
                    fragments=fragments,
                    model=model,
                )
                _merge_category_result(summary, config, result, fragments)
                break
            except OllamaRetryableError as exc:
                last_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Retrying Ollama category analysis after retryable error "
                        "for %s, attempt %s/%s: %s",
                        config.name,
                        attempt + 1,
                        max_attempts,
                        exc.detail,
                    )
        else:
            raise OllamaError(
                f"Ollama returned invalid response for {config.name} after 2 attempts: "
                f"{last_error.detail if last_error else 'unknown retryable error'}"
            ) from last_error

    return summary


async def analyze_category(
    config: CategoryConfig,
    fragments: list[PdfTextFragment],
    model: str | None = None,
) -> dict[str, Any]:
    prompt = _build_category_prompt(config, fragments)
    return await _request_category_summary(
        prompt=prompt,
        config=config,
        model=model,
    )


async def _request_category_summary(
    prompt: str,
    config: CategoryConfig,
    model: str | None = None,
) -> dict[str, Any]:
    try:
        selected_model = model or get_ollama_model()
        base_url = get_ollama_base_url()
        timeout_seconds = get_ollama_timeout_seconds()
        temperature = get_ollama_temperature()
        configured_num_predict = get_ollama_num_predict()
    except SettingsError as exc:
        raise OllamaError(str(exc), 500) from exc

    payload = {
        "model": selected_model,
        "stream": False,
        "format": _build_category_json_schema(config),
        "messages": [
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "Ты анализируешь российские государственные контракты. "
                    "Отвечай строго валидным JSON без markdown и без рассуждений. "
                    "Запрещено додумывать: используй только факты из переданных фрагментов. "
                    "Поле value всегда заполняй на русском языке. Не переводи данные на английский."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "num_predict": min(configured_num_predict, config.num_predict),
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
            f"Ollama timed out while analyzing {config.name} after {timeout_seconds} seconds. "
            "Reduce OLLAMA_MAX_FRAGMENT_CHARS, OLLAMA_MAX_FRAGMENTS_PER_CATEGORY, "
            "or use a smaller local model.",
            504,
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {500, 502, 503}:
            raise OllamaTransientHTTPError(
                f"Ollama returned temporary HTTP {status_code} while analyzing {config.name}"
            ) from exc
        raise OllamaError(
            f"Ollama returned HTTP {status_code} while analyzing {config.name}"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed while analyzing {config.name}") from exc

    try:
        content = response.json()["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaResponseFormatError(
            f"Ollama returned an unexpected response for {config.name}"
        ) from exc

    return _parse_category_summary(content, config)


def _build_category_json_schema(config: CategoryConfig) -> dict[str, Any]:
    if config.max_results == 1:
        field_schema = {
            "anyOf": [
                {"type": "null"},
                {"$ref": "#/$defs/summary_item"},
            ],
        }
    else:
        field_schema = {
            "type": "array",
            "maxItems": config.max_results,
            "items": {"$ref": "#/$defs/summary_item"},
        }

    return {
        "type": "object",
        "properties": {
            config.response_field: field_schema,
        },
        "required": [config.response_field],
        "additionalProperties": False,
        "$defs": {
            "summary_item": MODEL_ITEM_JSON_SCHEMA,
        },
    }


def _build_category_prompt(config: CategoryConfig, fragments: list[PdfTextFragment]) -> str:
    return (
        "/no_think\n"
        f"Анализируй только категорию: {config.name}.\n"
        f"Задача: {config.description}\n"
        f"Верни не больше {config.max_results} результат(ов).\n"
        "Поле value пиши на русском языке. Не переводи текст договора на другой язык. "
        "Если формулировка есть во фрагменте, сохраняй ее максимально близко к оригиналу.\n"
        "Для каждого найденного значения обязательно укажи fragment_ids: список id фрагментов, "
        "из которых взят факт. Для fragment_ids бери только значение после fragment= "
        "в заголовке фрагмента, например clause-2.1-p3-3. "
        "Не возвращай clause и pages: их заполнит программа по fragment_id.\n"
        f"Если категория не найдена в переданных фрагментах, верни {config.empty_value} "
        f"для поля {config.response_field}.\n"
        "Верни JSON строго такой структуры:\n"
        f"{_category_json_example(config)}\n\n"
        f"Фрагменты договора:\n{_format_fragments(config.name, fragments)}"
    )


def _category_json_example(config: CategoryConfig) -> str:
    if config.max_results == 1:
        return (
            "{\n"
            f'  "{config.response_field}": {{\n'
            '    "value": "сумма контракта как указано во фрагментах",\n'
            '    "fragment_ids": ["fragment-id-from-header"]\n'
            "  }\n"
            "}"
        )

    return (
        "{\n"
        f'  "{config.response_field}": [\n'
        '    {"value": "факт как указано во фрагментах", '
        '"fragment_ids": ["fragment-id-from-header"]}\n'
        "  ]\n"
        "}"
    )


def _format_fragments(category: str, fragments: list[PdfTextFragment]) -> str:
    if not fragments:
        return f"## {category}\nФрагменты не найдены."

    sections = [f"## {category}"]
    for fragment in fragments:
        pages = ", ".join(str(page) for page in fragment.page_numbers)
        sections.append(
            f"[fragment={fragment.fragment_id}; clause={fragment.clause or 'null'}; "
            f"pages=[{pages}]; score={fragment.score}]\n"
            f"{fragment.text}"
        )

    return "\n\n---\n\n".join(sections)


def _parse_category_summary(content: str, config: CategoryConfig) -> dict[str, Any]:
    if not content.strip():
        raise OllamaResponseFormatError(f"Ollama returned empty response for {config.name}")

    data = _json_loads_lenient(content, config.name)
    if config.response_field not in data:
        raise OllamaResponseFormatError(
            f"Ollama JSON response for {config.name} must contain {config.response_field}"
        )

    try:
        ContractSummary.model_validate(_summary_data_from_category(config, data))
    except ValidationError as exc:
        raise OllamaResponseFormatError(
            f"Ollama returned invalid schema for {config.name}: {exc}"
        ) from exc
    return data


def _summary_data_from_category(config: CategoryConfig, data: dict[str, Any]) -> dict[str, Any]:
    value = data.get(config.response_field)
    if config.name == "amount":
        contract_amount = _model_item_to_summary_item(value, {})
    else:
        contract_amount = None

    return {
        "contract_amount": contract_amount,
        "performance_terms": (
            [_model_item_to_summary_item(item, {}) for item in value]
            if config.name == "terms" and isinstance(value, list)
            else []
        ),
        "contractor_requirements": (
            [_model_item_to_summary_item(item, {}) for item in value]
            if config.name == "requirements" and isinstance(value, list)
            else []
        ),
        "penalties": (
            [_model_item_to_summary_item(item, {}) for item in value]
            if config.name == "penalties" and isinstance(value, list)
            else []
        ),
    }


def _merge_category_result(
    summary: ContractSummary,
    config: CategoryConfig,
    result: dict[str, Any],
    fragments: list[PdfTextFragment],
) -> None:
    value = result.get(config.response_field)
    fragment_lookup = {fragment.fragment_id: fragment for fragment in fragments}

    if config.name == "amount":
        summary.contract_amount = _model_item_to_summary_item(
            value,
            fragment_lookup,
            category=config.name,
            require_known_sources=True,
        )
        return

    items = getattr(summary, config.response_field)
    if not isinstance(value, list):
        return

    items.extend(
        item
        for item in (
            _model_item_to_summary_item(
                model_item,
                fragment_lookup,
                category=config.name,
                require_known_sources=True,
            )
            for model_item in value[: config.max_results]
        )
        if item is not None
    )


def _model_item_to_summary_item(
    model_item: Any,
    fragment_lookup: dict[str, PdfTextFragment],
    category: str | None = None,
    require_known_sources: bool = False,
) -> SummaryItem | None:
    if not isinstance(model_item, dict):
        return None

    value = str(model_item.get("value", "")).strip()
    if not value:
        return None

    fragment_ids = model_item.get("fragment_ids", [])
    if not isinstance(fragment_ids, list):
        fragment_ids = [fragment_ids]

    sources = []
    unknown_fragment_ids = []
    seen_sources = set()
    for fragment_id in fragment_ids:
        fragment = fragment_lookup.get(str(fragment_id))
        if fragment is None:
            unknown_fragment_ids.append(str(fragment_id))
            continue

        key = (fragment.clause, fragment.page_numbers)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        sources.append(
            {
                "clause": fragment.clause,
                "pages": list(fragment.page_numbers),
            }
        )

    if require_known_sources and not sources:
        category_detail = f" for {category}" if category else ""
        detail = (
            f": {', '.join(unknown_fragment_ids)}"
            if unknown_fragment_ids
            else ": no fragment_ids"
        )
        raise OllamaResponseFormatError(
            "Ollama returned unknown fragment_ids"
            f"{category_detail}{detail}"
        )

    return SummaryItem.model_validate({"value": value, "sources": sources})


def _json_loads_lenient(content: str, category: str) -> dict[str, Any]:
    try:
        return _ensure_json_object(json.loads(content), category)
    except json.JSONDecodeError as original_exc:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise OllamaResponseFormatError(
                f"Ollama did not return JSON for {category}"
            ) from original_exc
        try:
            return _ensure_json_object(json.loads(match.group(0)), category)
        except json.JSONDecodeError as extracted_exc:
            raise OllamaResponseFormatError(
                f"Ollama returned malformed JSON for {category}. Increase OLLAMA_NUM_PREDICT "
                "or reduce OLLAMA_MAX_FRAGMENTS_PER_CATEGORY / OLLAMA_MAX_FRAGMENT_CHARS."
            ) from extracted_exc


def _ensure_json_object(loaded: Any, category: str) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        raise OllamaResponseFormatError(
            f"Ollama JSON response for {category} must be an object"
        )

    return loaded
