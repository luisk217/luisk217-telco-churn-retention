"""Definición de modelos candidatos y del preprocesamiento."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.data import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("bin", "passthrough", BINARY_COLUMNS),
            # handle_unknown="ignore": una categoría nueva no rompe la predicción.
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=True,
    )


def candidate_models(random_state: int) -> dict[str, Pipeline]:
    """Línea base interpretable frente a un modelo de boosting.

    No se reponderan las clases: el desbalance se resuelve con el umbral
    económico, y así las probabilidades quedan calibradas para calcular
    el valor esperado de cada contacto.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
    }


def readable_feature(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    prefix, _, rest = name.partition("__")
    if prefix == "cat":
        for col in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
            if rest.startswith(col + "_"):
                return f"{col} = {rest[len(col) + 1 :]}"
    return rest or name
