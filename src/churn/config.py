"""Carga y validación de la configuración del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """La configuración no es válida."""


@dataclass(frozen=True)
class DataConfig:
    url: str
    sha256: str
    max_download_bytes: int
    raw_path: Path
    db_path: Path


@dataclass(frozen=True)
class ModelConfig:
    random_state: int
    test_size: float
    val_size: float
    cv_folds: int
    artifacts_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class EconomicsConfig:
    horizon_months: int
    retention_success_rate: float
    cost_per_contact: float


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    economics: EconomicsConfig


def _resolve(path: str, root: Path) -> Path:
    """Resuelve rutas relativas a la raíz e impide que salgan de ella."""
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(f"La ruta '{path}' sale del directorio del proyecto.")
    return resolved


def _validate(cfg: Config) -> None:
    if urlparse(cfg.data.url).scheme != "https":
        raise ConfigError("La URL de datos debe usar HTTPS.")
    if len(cfg.data.sha256) != 64 or not all(c in "0123456789abcdef" for c in cfg.data.sha256):
        raise ConfigError("sha256 debe ser un hash hexadecimal de 64 caracteres en minúscula.")
    if cfg.data.max_download_bytes <= 0:
        raise ConfigError("max_download_bytes debe ser positivo.")
    if not 0 < cfg.model.test_size < 1 or not 0 < cfg.model.val_size < 1:
        raise ConfigError("test_size y val_size deben estar entre 0 y 1.")
    if cfg.model.test_size + cfg.model.val_size >= 1:
        raise ConfigError("test_size + val_size debe ser menor que 1.")
    if cfg.model.cv_folds < 2:
        raise ConfigError("cv_folds debe ser al menos 2.")
    if cfg.economics.horizon_months <= 0:
        raise ConfigError("horizon_months debe ser positivo.")
    if not 0 < cfg.economics.retention_success_rate <= 1:
        raise ConfigError("retention_success_rate debe estar en (0, 1].")
    if cfg.economics.cost_per_contact < 0:
        raise ConfigError("cost_per_contact no puede ser negativo.")


def load_config(path: Path = DEFAULT_CONFIG_PATH, root: Path = PROJECT_ROOT) -> Config:
    """Lee el YAML con `safe_load` (nunca `load`) y valida cada valor."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        d, m, e = raw["data"], raw["model"], raw["economics"]
        cfg = Config(
            data=DataConfig(
                url=str(d["url"]),
                sha256=str(d["sha256"]).lower(),
                max_download_bytes=int(d["max_download_bytes"]),
                raw_path=_resolve(d["raw_path"], root),
                db_path=_resolve(d["db_path"], root),
            ),
            model=ModelConfig(
                random_state=int(m["random_state"]),
                test_size=float(m["test_size"]),
                val_size=float(m["val_size"]),
                cv_folds=int(m["cv_folds"]),
                artifacts_dir=_resolve(m["artifacts_dir"], root),
                reports_dir=_resolve(m["reports_dir"], root),
            ),
            economics=EconomicsConfig(
                horizon_months=int(e["horizon_months"]),
                retention_success_rate=float(e["retention_success_rate"]),
                cost_per_contact=float(e["cost_per_contact"]),
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Falta o es inválido el campo de configuración: {exc}") from exc
    _validate(cfg)
    return cfg
