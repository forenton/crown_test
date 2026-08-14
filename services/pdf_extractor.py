from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


RELEVANT_KEYWORDS = (
    "цена",
    "стоимость",
    "сумма",
    "руб",
    "срок",
    "поставка",
    "выполнение",
    "исполнитель",
    "поставщик",
    "подрядчик",
    "требован",
    "штраф",
    "пеня",
    "пени",
    "неустой",
    "ответствен",
)
KEYWORD_GROUPS = {
    "amount": ("цена", "стоимость", "сумма", "руб", "коп"),
    "terms": ("срок", "поставка", "выполнение", "оказание", "этап"),
    "requirements": ("исполнитель", "поставщик", "подрядчик", "требован", "обязан", "лиценз"),
    "penalties": ("штраф", "пеня", "пени", "неустой", "ответствен"),
}
CATEGORY_PHRASE_WEIGHTS = {
    "amount": {
        "цена контракта": 8,
        "цена настоящего контракта": 8,
        "составляет": 2,
    },
    "terms": {
        "срок исполнения контракта": 18,
        "срок исполнения настоящего контракта": 18,
        "срок выполнения работ": 12,
        "срок выполнения": 10,
        "срок поставки": 10,
        "срок оказания услуг": 10,
        "поставка товара осуществляется в срок": 18,
        "поставка товара осуществляется": 8,
        "осуществляется в срок": 14,
        "место и срок поставки": 16,
        "сроки поставки": 12,
        "сроки выполнения": 12,
        "график выполнения обязательств": 22,
        "график выполнения": 18,
        "сведения об обязательствах сторон": 8,
        "сроки, порядок и место поставки": 12,
        "поставка товара должна осуществляться в сроки": 18,
        "установленные сроки": 8,
        "до 31 декабря": 6,
        "до 30 сентября": 6,
        "в течение": 5,
        "календарных дней": 5,
        "рабочих дней": 4,
        "даты заключения контракта": 6,
        "допускается досрочное": 5,
        "работы выполняются одним этапом": 6,
    },
    "requirements": {
        "подрядчик обязан": 6,
        "поставщик обязан": 6,
        "исполнитель обязан": 6,
        "требования к": 5,
        "должны соответствовать": 4,
    },
    "penalties": {
        "штраф": 5,
        "пеня начисляется": 8,
        "неустой": 5,
        "ответственность сторон": 8,
    },
}
CATEGORY_NEGATIVE_WEIGHTS = {
    "terms": {
        "за каждый факт": 25,
        "размер штрафа": 25,
        "штраф устанавливается": 25,
        "пеня": 20,
        "пени": 20,
        "неустой": 20,
        "просрочки исполнения": 12,
        "обстоятельств непреодолимой силы": 14,
        "непреодолимой силы": 14,
        "претензи": 10,
        "досудебный порядок": 12,
        "приемк": 8,
        "недостатк": 8,
        "документ о приемке": 10,
        "изменения юридических адресов": 12,
        "изменении реквизитов": 12,
        "возвратить заказчику": 10,
        "цена контракта является твердой": 25,
        "определяется на весь срок исполнения": 22,
        "сроки оплаты": 12,
        "порядок и сроки оплаты": 14,
        "конфиденциальную информацию": 12,
        "гарантийный срок": 10,
        "срок действия независимой гарантии": 12,
        "обеспечение исполнения контракта": 8,
        "неустоек": 4,
        "пеней": 4,
    },
}
CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})[.)]\s+")
DEFAULT_MAX_FRAGMENT_CHARS = 6_000
FRAGMENT_OVERLAP_CHARS = 1_000


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class PdfTextFragment:
    fragment_id: str
    clause: str | None
    page_numbers: tuple[int, ...]
    text: str
    score: int


class PdfTextExtractionError(Exception):
    def __init__(self, detail: str, status_code: int = 400) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def extract_pdf_pages(content: bytes) -> dict[str, object]:
    try:
        reader = PdfReader(BytesIO(content))
        pages = []

        for index, page in enumerate(reader.pages, start=1):
            text = normalize_pdf_text(page.extract_text() or "")
            if not text:
                continue

            pages.append(PdfPageText(page_number=index, text=text))
    except (PdfReadError, ValueError, TypeError, OSError) as exc:
        raise PdfTextExtractionError("Could not extract text from PDF") from exc

    if not pages:
        raise PdfTextExtractionError(
            "PDF has no extractable text. OCR is required for scanned documents."
        )

    return {
        "pages": pages,
        "page_count": len(reader.pages),
    }


def build_relevant_fragments(
    pages: list[PdfPageText],
    max_fragments_per_category: int,
    max_fragment_chars: int = DEFAULT_MAX_FRAGMENT_CHARS,
) -> dict[str, list[PdfTextFragment]]:
    if not pages:
        return {category: [] for category in KEYWORD_GROUPS}

    fragments = split_pages_into_clauses(pages)
    if not fragments:
        fragments = build_page_windows(pages, max_fragment_chars=max_fragment_chars)
    else:
        fragments = split_oversized_fragments(
            fragments,
            max_fragment_chars=max_fragment_chars,
            overlap_chars=FRAGMENT_OVERLAP_CHARS,
        )

    result: dict[str, list[PdfTextFragment]] = {}
    for category, keywords in KEYWORD_GROUPS.items():
        scored = [
            _with_score(fragment, category_fragment_score(category, fragment.text, keywords))
            for fragment in fragments
        ]
        relevant = [fragment for fragment in scored if fragment.score > 0]
        relevant.sort(key=lambda item: (-item.score, item.page_numbers, item.fragment_id))
        result[category] = relevant[: max(1, max_fragments_per_category)]

    return result


def split_pages_into_clauses(pages: list[PdfPageText]) -> list[PdfTextFragment]:
    fragments: list[PdfTextFragment] = []
    current_lines: list[str] = []
    current_pages: set[int] = set()
    current_number: str | None = None
    current_section: str | None = None

    for page in pages:
        for line in page.text.splitlines():
            if is_section_header(line) and current_lines:
                fragments.append(
                    _make_fragment(current_number, current_section, current_pages, current_lines)
                )
                current_lines = []
                current_pages = set()
                current_number = None

            current_section = update_section_context(line, current_section)
            clause_match = CLAUSE_NUMBER_RE.match(line)
            if clause_match and current_lines:
                fragments.append(
                    _make_fragment(current_number, current_section, current_pages, current_lines)
                )
                current_lines = []
                current_pages = set()

            if clause_match:
                current_number = clause_match.group(1)

            current_lines.append(line)
            current_pages.add(page.page_number)

    if current_lines:
        fragments.append(_make_fragment(current_number, current_section, current_pages, current_lines))

    return [fragment for fragment in fragments if len(fragment.text) >= 20]


def build_page_windows(
    pages: list[PdfPageText],
    max_fragment_chars: int,
) -> list[PdfTextFragment]:
    fragments: list[PdfTextFragment] = []
    current_text = ""
    current_pages: list[int] = []

    for page in pages:
        page_text = f"[page={page.page_number}]\n{page.text}"
        if current_text and len(current_text) + len(page_text) > max_fragment_chars:
            fragments.append(_make_window_fragment(current_pages, current_text))
            current_text = current_text[-FRAGMENT_OVERLAP_CHARS:]
            current_pages = current_pages[-1:] if current_pages else []

        current_text = f"{current_text}\n\n{page_text}".strip()
        if page.page_number not in current_pages:
            current_pages.append(page.page_number)

    if current_text and current_pages:
        fragments.append(_make_window_fragment(current_pages, current_text))

    return fragments


def split_oversized_fragments(
    fragments: list[PdfTextFragment],
    max_fragment_chars: int,
    overlap_chars: int,
) -> list[PdfTextFragment]:
    result: list[PdfTextFragment] = []

    for fragment in fragments:
        if len(fragment.text) <= max_fragment_chars:
            result.append(fragment)
            continue

        start = 0
        part_number = 1
        step = max(1, max_fragment_chars - overlap_chars)
        while start < len(fragment.text):
            chunk = fragment.text[start : start + max_fragment_chars]
            result.append(
                PdfTextFragment(
                    fragment_id=f"{fragment.fragment_id}-part-{part_number}",
                    clause=fragment.clause,
                    page_numbers=fragment.page_numbers,
                    text=chunk.strip(),
                    score=document_relevance_score(chunk),
                )
            )
            if start + max_fragment_chars >= len(fragment.text):
                break
            start += step
            part_number += 1

    return result


def normalize_pdf_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    normalized_lines = []

    for line in lines:
        if not line:
            if normalized_lines and normalized_lines[-1]:
                normalized_lines.append("")
            continue
        normalized_lines.append(re.sub(r"\s+", " ", line))

    return "\n".join(normalized_lines).strip()


def _make_fragment(
    clause_number: str | None,
    section_context: str | None,
    page_numbers: set[int],
    lines: list[str],
) -> PdfTextFragment:
    ordered_pages = tuple(sorted(page_numbers))
    fragment_id = f"clause-{clause_number or 'intro'}-p{ordered_pages[0]}-{ordered_pages[-1]}"
    text = "\n".join(lines).strip()
    return PdfTextFragment(
        fragment_id=fragment_id,
        clause=format_clause_label(clause_number, section_context),
        page_numbers=ordered_pages,
        text=text,
        score=document_relevance_score(text),
    )


def _make_window_fragment(page_numbers: list[int], text: str) -> PdfTextFragment:
    return PdfTextFragment(
        fragment_id=f"pages-{page_numbers[0]}-{page_numbers[-1]}",
        clause=None,
        page_numbers=tuple(page_numbers),
        text=text.strip(),
        score=document_relevance_score(text),
    )


def _with_score(fragment: PdfTextFragment, score: int) -> PdfTextFragment:
    return PdfTextFragment(
        fragment_id=fragment.fragment_id,
        clause=fragment.clause,
        page_numbers=fragment.page_numbers,
        text=fragment.text,
        score=score,
    )


def update_section_context(line: str, current_section: str | None) -> str | None:
    normalized = line.strip().lower()
    if not normalized:
        return current_section

    appendix_match = re.match(r"приложение\s*№\s*(\d+)", normalized)
    if appendix_match:
        return f"Приложение № {appendix_match.group(1)}"

    if "техническое задание" in normalized:
        appendix = current_section if current_section and current_section.startswith("Приложение") else None
        return f"{appendix} / Техническое задание" if appendix else "Техническое задание"

    if "локальный сметный расчет" in normalized or "смета" in normalized:
        appendix = current_section if current_section and current_section.startswith("Приложение") else None
        return f"{appendix} / Локальный сметный расчет" if appendix else "Локальный сметный расчет"

    return current_section


def is_section_header(line: str) -> bool:
    normalized = line.strip().lower()
    return (
        bool(re.match(r"приложение\s*№\s*\d+", normalized))
        or "техническое задание" in normalized
        or normalized.startswith("локальный сметный расчет")
        or normalized.startswith("смета")
    )


def format_clause_label(clause_number: str | None, section_context: str | None) -> str | None:
    if not clause_number:
        return section_context
    if section_context:
        return f"{section_context}, пункт {clause_number}"
    return clause_number


def document_relevance_score(text: str) -> int:
    lower_text = text.lower()
    return sum(lower_text.count(keyword) for keyword in RELEVANT_KEYWORDS)


def fragment_keyword_score(text: str, keywords: tuple[str, ...]) -> int:
    lower_text = text.lower()
    return sum(lower_text.count(keyword) for keyword in keywords)


def category_fragment_score(category: str, text: str, keywords: tuple[str, ...]) -> int:
    lower_text = text.lower()
    score = fragment_keyword_score(lower_text, keywords)

    for phrase, weight in CATEGORY_PHRASE_WEIGHTS.get(category, {}).items():
        score += lower_text.count(phrase) * weight

    for phrase, weight in CATEGORY_NEGATIVE_WEIGHTS.get(category, {}).items():
        score -= lower_text.count(phrase) * weight

    if category == "terms":
        score += main_performance_term_score(lower_text)

    return max(0, score)


def main_performance_term_score(lower_text: str) -> int:
    score = 0

    main_term_patterns = (
        r"срок\s+(?:исполнения|поставки|выполнения|оказания)[^.]{0,120}\bдо\s+\d{1,2}\s+[а-яё]+\s+\d{4}",
        r"срок\s+(?:исполнения|поставки|выполнения|оказания)[^.]{0,120}\bдо\s+\d{1,2}[.]\d{1,2}[.]\d{4}",
        r"(?:поставка|работы|услуги)[^.]{0,80}осуществля[а-яё]+\s+в\s+срок\s+до\s+",
        r"в\s+течение\s+\d+[^.]{0,80}(?:дней|рабочих дней|календарных дней)[^.]{0,80}даты\s+заключения\s+контракта",
    )
    for pattern in main_term_patterns:
        score += len(re.findall(pattern, lower_text)) * 18

    secondary_term_patterns = (
        r"в\s+течение\s+\d+[^.]{0,80}(?:получения|обнаружения|подписания|расторжения)",
        r"обязан[а-яё\s]{0,80}в\s+течение\s+\d+",
    )
    for pattern in secondary_term_patterns:
        score -= len(re.findall(pattern, lower_text)) * 8

    return score
