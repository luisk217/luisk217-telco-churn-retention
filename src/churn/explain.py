"""Explicaciones SHAP globales y por cliente."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churn.data import FEATURE_COLUMNS
from churn.modeling import readable_feature


def shap_values(model: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Valores SHAP (en log-odds) sobre las variables ya transformadas."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = [readable_feature(n) for n in prep.get_feature_names_out()]
    X_t = prep.transform(X[FEATURE_COLUMNS])
    bg_t = prep.transform(background[FEATURE_COLUMNS])
    if hasattr(clf, "get_booster"):
        explainer = shap.TreeExplainer(clf, data=bg_t, model_output="raw")
    else:
        explainer = shap.LinearExplainer(clf, bg_t)
    values = explainer.shap_values(X_t)
    return pd.DataFrame(np.asarray(values), columns=names, index=X.index)


def global_importance(values: pd.DataFrame) -> pd.DataFrame:
    return (
        values.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .rename_axis("feature")
        .reset_index()
    )


def top_reasons(values: pd.DataFrame, k: int = 3) -> pd.Series:
    """Las k variables que más empujan a cada cliente hacia la fuga."""

    def _row(row: pd.Series) -> str:
        positive = row[row > 0].nlargest(k)
        return "; ".join(positive.index) if not positive.empty else "-"

    return values.apply(_row, axis=1).rename("top_reasons")
