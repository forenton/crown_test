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
        "format": "json",
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
        "Если поле не найдено в переданных фрагментах, верни null для суммы или пустой список. "
        "Не используй типовые формулировки, которых нет во фрагментах.\n"
        "Верни JSON строго такой структуры:\n"
        "{\n"
        '  "contract_amount": "сумма контракта строкой или null",\n'
        '  "performance_terms": ["сроки выполнения, поставки, этапы"],\n'
        '  "contractor_requirements": ["ключевые требования к исполнителю"],\n'
        '  "penalties": ["штрафы, пени, ответственность"]\n'
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
                f"[fragment={fragment.fragment_id}; pages={pages}; score={fragment.score}]\n"
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
        loaded = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise OllamaError("Ollama did not return JSON")
        loaded = json.loads(match.group(0))

    if not isinstance(loaded, dict):
        raise OllamaError("Ollama JSON response must be an object")

    return loaded
