"""Lectura y escritura seguras de archivos que tocan usuarios."""

from __future__ import annotations

import io

import pandas as pd

from churn.data import load_raw

MAX_UPLOAD_BYTES = 5_000_000
# Prefijos que Excel/LibreOffice interpretan como fórmula (CSV injection, OWASP).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_formula(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def safe_csv_export(df: pd.DataFrame) -> bytes:
    """CSV en UTF-8 con celdas de texto neutralizadas contra inyección de fórmulas."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    safe = df.copy()
    for col in text_cols:
        safe[col] = safe[col].map(neutralize_formula)
    return safe.to_csv(index=False).encode("utf-8")


def read_uploaded_csv(data: bytes) -> pd.DataFrame:
    """Lee un CSV subido por un usuario con límite de tamaño y sin inferir tipos."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"El archivo supera {MAX_UPLOAD_BYTES // 1_000_000} MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("El archivo debe estar codificado en UTF-8.") from exc
    return load_raw(io.StringIO(text))
