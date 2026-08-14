from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


class SettingsError(Exception):
    pass


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise SettingsError(f"Environment variable {name} is required")
    return value.strip()


def get_required_int_env(name: str) -> int:
    raw_value = get_required_env(name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Environment variable {name} must be an integer") from exc


def get_required_float_env(name: str) -> float:
    raw_value = get_required_env(name)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Environment variable {name} must be a number") from exc


def get_ollama_base_url() -> str:
    return get_required_env("OLLAMA_BASE_URL").rstrip("/")


def get_ollama_model() -> str:
    return get_required_env("OLLAMA_MODEL")


def get_ollama_timeout_seconds() -> int:
    return get_required_int_env("OLLAMA_TIMEOUT_SECONDS")


def get_ollama_temperature() -> float:
    return get_required_float_env("OLLAMA_TEMPERATURE")


def get_ollama_max_fragment_chars() -> int:
    return get_required_int_env("OLLAMA_MAX_FRAGMENT_CHARS")


def get_ollama_max_fragments_per_category() -> int:
    return get_required_int_env("OLLAMA_MAX_FRAGMENTS_PER_CATEGORY")


def get_ollama_num_predict() -> int:
    return get_required_int_env("OLLAMA_NUM_PREDICT")
