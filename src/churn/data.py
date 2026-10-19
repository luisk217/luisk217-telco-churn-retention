"""Descarga verificada, validación de esquema y limpieza del dataset."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import IO

import pandas as pd

from churn.config import DataConfig

logger = logging.getLogger(__name__)

ID_COLUMN = "customerID"
TARGET = "Churn"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLUMNS = ["SeniorCitizen"]
# Valores permitidos por columna categórica: cualquier otro valor se rechaza.
CATEGORICAL_DOMAINS: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_DOMAINS)
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
EXPECTED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]


class DataValidationError(ValueError):
    """Los datos no cumplen el esquema esperado."""


class IntegrityError(RuntimeError):
    """El archivo descargado no coincide con la huella esperada."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(cfg: DataConfig, timeout: float = 30.0) -> Path:
    """Descarga el CSV por HTTPS con límite de tamaño y verificación SHA-256.

    Se escribe primero en un temporal y solo se mueve al destino si el hash
    coincide, así nunca queda en disco un archivo a medias o manipulado.
    """
    if cfg.raw_path.exists() and sha256_of(cfg.raw_path) == cfg.sha256:
        logger.info("Dataset ya presente y verificado: %s", cfg.raw_path)
        return cfg.raw_path

    cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
    # La URL se valida como HTTPS al cargar la configuración (ver churn.config).
    request = urllib.request.Request(  # noqa: S310
        cfg.url, headers={"User-Agent": "telco-churn-retention"}
    )
    fd, tmp_name = tempfile.mkstemp(dir=cfg.raw_path.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(request, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:  # noqa: S310  # nosec B310
            while chunk := resp.read(65536):
                received += len(chunk)
                if received > cfg.max_download_bytes:
                    raise IntegrityError("La descarga supera el tamaño máximo permitido.")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != cfg.sha256:
            raise IntegrityError(
                f"SHA-256 inesperado ({digest.hexdigest()}). La fuente cambió o fue alterada."
            )
        tmp_path.replace(cfg.raw_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("Dataset descargado y verificado: %s", cfg.raw_path)
    return cfg.raw_path


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Comprueba columnas, dominios categóricos, rangos y unicidad del ID."""
    expected = EXPECTED_COLUMNS if require_target else [ID_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas: {missing}")
    if df.empty:
        raise DataValidationError("El dataset está vacío.")
    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise DataValidationError(f"{ID_COLUMN} tiene valores nulos o duplicados.")
    for col, domain in CATEGORICAL_DOMAINS.items():
        invalid = set(df[col].dropna().astype(str).unique()) - domain
        if invalid or df[col].isna().any():
            raise DataValidationError(f"Valores no permitidos en {col}: {sorted(invalid)[:5]}")
    if not set(pd.unique(df["SeniorCitizen"])) <= {0, 1}:
        raise DataValidationError("SeniorCitizen debe ser 0 o 1.")
    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise DataValidationError(f"{col} tiene valores nulos, no numéricos o negativos.")
    if require_target and not set(df[TARGET].unique()) <= {0, 1}:
        raise DataValidationError(f"{TARGET} debe ser binario (0/1) tras la limpieza.")


def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Normaliza tipos y corrige el caso conocido de TotalCharges vacío.

    En la fuente, 11 clientes con tenure == 0 tienen TotalCharges en blanco:
    aún no han pagado ninguna factura, así que el valor correcto es 0.
    """
    out = df.copy()
    for col in CATEGORICAL_COLUMNS + [ID_COLUMN]:
        if col in out:
            out[col] = out[col].astype(str).str.strip()
    total = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank_new_customers = total.isna() & (pd.to_numeric(out["tenure"], errors="coerce") == 0)
    out["TotalCharges"] = total.mask(blank_new_customers, 0.0)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce")
    if require_target and TARGET in out:
        out[TARGET] = out[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    validate_schema(out, require_target=require_target)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    if require_target:
        out[TARGET] = out[TARGET].astype(int)
    return out


def load_raw(path: Path | IO[str]) -> pd.DataFrame:
    # dtype=str evita inferencias silenciosas; los tipos se fijan en `clean`.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
