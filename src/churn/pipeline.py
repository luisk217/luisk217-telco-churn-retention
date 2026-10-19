"""Pipeline de punta a punta: datos -> SQLite -> modelo -> umbral económico -> reportes."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn import __version__
from churn.artifacts import load_model, save_model
from churn.config import Config, EconomicsConfig
from churn.data import FEATURE_COLUMNS, ID_COLUMN, TARGET, clean, download_dataset, load_raw
from churn.db import read_customers, write_customers
from churn.economics import (
    baselines,
    best_threshold,
    customer_value,
    evaluate_threshold,
    expected_value,
    profit_curve,
)
from churn.explain import global_importance, shap_values, top_reasons
from churn.modeling import candidate_models

logger = logging.getLogger(__name__)


def prepare_data(cfg: Config) -> pd.DataFrame:
    raw_path = download_dataset(cfg.data)
    df = clean(load_raw(raw_path))
    write_customers(df, cfg.data.db_path)
    logger.info("%d clientes cargados en %s", len(df), cfg.data.db_path)
    return read_customers(cfg.data.db_path)


def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """División estratificada en entrenamiento, validación y prueba."""
    m = cfg.model
    train_val, test = train_test_split(
        df, test_size=m.test_size, stratify=df[TARGET], random_state=m.random_state
    )
    val_fraction = m.val_size / (1 - m.test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val[TARGET], random_state=m.random_state
    )
    return train, val, test


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
    }


def select_model(train: pd.DataFrame, cfg: Config) -> tuple[str, Pipeline, dict[str, float]]:
    """Elige el modelo por PR-AUC en validación cruzada (sensible al desbalance)."""
    X, y = train[FEATURE_COLUMNS], train[TARGET]
    cv = StratifiedKFold(cfg.model.cv_folds, shuffle=True, random_state=cfg.model.random_state)
    scores: dict[str, float] = {}
    candidates = candidate_models(cfg.model.random_state)
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
        scores[name] = round(float(cv_scores.mean()), 4)
        logger.info("CV PR-AUC %s: %.4f ± %.4f", name, cv_scores.mean(), cv_scores.std())
    best = max(scores, key=scores.__getitem__)
    return best, candidates[best].fit(X, y), scores


def sensitivity_analysis(
    y_val: np.ndarray,
    p_val: np.ndarray,
    m_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    m_test: np.ndarray,
    base: EconomicsConfig,
) -> pd.DataFrame:
    """¿Sigue valiendo la pena el modelo si los supuestos de negocio cambian?

    Para cada escenario se recalcula el umbral en validación y se mide en prueba.
    """
    rows = []
    for cost in (10.0, 25.0, 50.0, 100.0):
        for success in (0.15, 0.30, 0.50):
            econ = EconomicsConfig(base.horizon_months, success, cost)
            t = best_threshold(y_val, p_val, customer_value(m_val, econ), econ).threshold
            v_test = customer_value(m_test, econ)
            result = evaluate_threshold(y_test, p_test, v_test, t, econ)
            ref = baselines(y_test, v_test, econ)
            rows.append(
                {
                    "cost_per_contact": cost,
                    "retention_success_rate": success,
                    "threshold": t,
                    "contacted": result.contacted,
                    "profit_model": round(result.profit, 2),
                    "profit_contact_everyone": round(ref["contact_everyone"], 2),
                    "share_of_oracle": round(result.profit / ref["perfect_oracle"], 3),
                }
            )
    return pd.DataFrame(rows)


def _plot_profit_curve(curve: pd.DataFrame, threshold: float, path) -> None:
    # Figure sin pyplot: no depende de un backend gráfico ni de estado global.
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(curve["threshold"], curve["profit"], color="#1f6feb")
    ax.axvline(
        threshold, color="#d1242f", linestyle="--", label=f"Umbral elegido = {threshold:.2f}"
    )
    ax.axhline(0, color="#57606a", linewidth=0.8)
    ax.set_xlabel("Umbral de probabilidad para contactar")
    ax.set_ylabel("Beneficio esperado (validación)")
    ax.set_title("Beneficio de la campaña según el umbral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _plot_importance(importance: pd.DataFrame, path, k: int = 15) -> None:
    top = importance.head(k).iloc[::-1]
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1f6feb")
    ax.set_xlabel("Media |SHAP| (impacto en log-odds)")
    ax.set_title("Variables que más influyen en la fuga")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def train(cfg: Config) -> dict[str, Any]:
    df = prepare_data(cfg)
    train_df, val_df, test_df = split(df, cfg)
    name, model, cv_scores = select_model(train_df, cfg)

    econ = cfg.economics
    val_proba = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_value = customer_value(val_df["MonthlyCharges"].to_numpy(), econ)
    chosen = best_threshold(val_df[TARGET].to_numpy(), val_proba, val_value, econ)

    # El conjunto de prueba solo se usa una vez, con el umbral ya fijado.
    y_test = test_df[TARGET].to_numpy()
    test_proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_value = customer_value(test_df["MonthlyCharges"].to_numpy(), econ)
    test_campaign = evaluate_threshold(y_test, test_proba, test_value, chosen.threshold, econ)
    test_baselines = baselines(y_test, test_value, econ)

    metrics = {
        "model": name,
        "cv_pr_auc": cv_scores,
        "threshold": chosen.threshold,
        "test_classification": classification_metrics(y_test, test_proba, chosen.threshold),
        "test_classification_at_0_5": classification_metrics(y_test, test_proba, 0.5),
        "test_campaign": {
            "customers": int(len(test_df)),
            "contacted": test_campaign.contacted,
            "churners_total": int(y_test.sum()),
            "churners_contacted": test_campaign.churners_contacted,
            "profit_model": round(test_campaign.profit, 2),
            "profit_at_0_5": round(
                evaluate_threshold(y_test, test_proba, test_value, 0.5, econ).profit, 2
            ),
            **{f"profit_{k}": round(v, 2) for k, v in test_baselines.items()},
        },
        "economics": econ.__dict__,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }

    reports = cfg.model.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    _plot_profit_curve(
        profit_curve(val_df[TARGET].to_numpy(), val_proba, val_value, econ),
        chosen.threshold,
        reports / "profit_curve.png",
    )
    background = train_df.sample(min(100, len(train_df)), random_state=cfg.model.random_state)
    importance = global_importance(shap_values(model, test_df, background))
    importance.to_csv(reports / "shap_importance.csv", index=False)
    _plot_importance(importance, reports / "shap_importance.png")
    sensitivity_analysis(
        val_df[TARGET].to_numpy(),
        val_proba,
        val_df["MonthlyCharges"].to_numpy(),
        y_test,
        test_proba,
        test_df["MonthlyCharges"].to_numpy(),
        econ,
    ).to_csv(reports / "sensitivity.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_model(
        model,
        {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "package_version": __version__,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "dataset_sha256": cfg.data.sha256,
            "model_name": name,
            "threshold": chosen.threshold,
            "features": FEATURE_COLUMNS,
        },
        cfg.model.artifacts_dir,
    )
    logger.info("Modelo %s guardado. Umbral óptimo: %.2f", name, chosen.threshold)
    return metrics


def score(df: pd.DataFrame, cfg: Config, with_reasons: bool = True) -> pd.DataFrame:
    """Lista de acción para retención, ordenada por valor esperado de contactar.

    ``df`` debe estar limpio y validado (ver ``churn.data.clean``).
    """
    model, manifest = load_model(cfg.model.artifacts_dir)
    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    value = customer_value(df["MonthlyCharges"].to_numpy(), cfg.economics)
    out = pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].to_numpy(),
            "churn_probability": np.round(proba, 4),
            "revenue_at_risk": np.round(value, 2),
            "expected_value_of_contact": np.round(expected_value(proba, value, cfg.economics), 2),
            "contact": proba >= manifest["threshold"],
        },
        index=df.index,
    )
    if with_reasons:
        background = df.sample(min(100, len(df)), random_state=cfg.model.random_state)
        out["top_reasons"] = top_reasons(shap_values(model, df, background))
    return out.sort_values("expected_value_of_contact", ascending=False).reset_index(drop=True)
