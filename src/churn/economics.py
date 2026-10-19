"""Traducción de probabilidades de fuga a decisiones y dinero.

Modelo de beneficio de una campaña de retención:

* Contactar a un cliente cuesta ``cost_per_contact`` (llamada + incentivo).
* Si el cliente iba a irse, la campaña lo retiene con probabilidad
  ``retention_success_rate`` y se conservan ``horizon_months`` de su
  ``MonthlyCharges``.
* Si el cliente no iba a irse, el contacto es un costo sin retorno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churn.config import EconomicsConfig


@dataclass(frozen=True)
class CampaignResult:
    threshold: float
    contacted: int
    churners_contacted: int
    profit: float


def customer_value(monthly_charges: np.ndarray, cfg: EconomicsConfig) -> np.ndarray:
    """Ingreso en riesgo por cliente durante el horizonte de análisis."""
    return np.asarray(monthly_charges, dtype=float) * cfg.horizon_months


def campaign_profit(
    y_true: np.ndarray, contact: np.ndarray, value: np.ndarray, cfg: EconomicsConfig
) -> float:
    """Beneficio esperado de contactar a los clientes marcados en ``contact``."""
    y_true, contact, value = (np.asarray(a) for a in (y_true, contact, value))
    contact = contact.astype(bool)
    saved = cfg.retention_success_rate * value[contact & (y_true == 1)].sum()
    cost = cfg.cost_per_contact * contact.sum()
    return float(saved - cost)


def evaluate_threshold(
    y_true: np.ndarray, proba: np.ndarray, value: np.ndarray, threshold: float, cfg: EconomicsConfig
) -> CampaignResult:
    contact = np.asarray(proba) >= threshold
    return CampaignResult(
        threshold=float(threshold),
        contacted=int(contact.sum()),
        churners_contacted=int((contact & (np.asarray(y_true) == 1)).sum()),
        profit=campaign_profit(y_true, contact, value, cfg),
    )


def profit_curve(
    y_true: np.ndarray, proba: np.ndarray, value: np.ndarray, cfg: EconomicsConfig
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = [evaluate_threshold(y_true, proba, value, t, cfg).__dict__ for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold(
    y_true: np.ndarray, proba: np.ndarray, value: np.ndarray, cfg: EconomicsConfig
) -> CampaignResult:
    """Umbral que maximiza el beneficio. Debe elegirse en validación, nunca en test."""
    curve = profit_curve(y_true, proba, value, cfg)
    best = curve.loc[curve["profit"].idxmax()]
    return CampaignResult(**best.to_dict())


def baselines(y_true: np.ndarray, value: np.ndarray, cfg: EconomicsConfig) -> dict[str, float]:
    """Escenarios de referencia para dimensionar el valor del modelo."""
    y_true = np.asarray(y_true)
    return {
        "no_campaign": 0.0,
        "contact_everyone": campaign_profit(y_true, np.ones_like(y_true), value, cfg),
        "perfect_oracle": campaign_profit(y_true, y_true == 1, value, cfg),
    }


def expected_value(proba: np.ndarray, value: np.ndarray, cfg: EconomicsConfig) -> np.ndarray:
    """Valor esperado de contactar a cada cliente: p * éxito * valor - costo."""
    return np.asarray(proba) * cfg.retention_success_rate * value - cfg.cost_per_contact
