from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from churn.config import load_config
from churn.data import CATEGORICAL_DOMAINS


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético con el mismo esquema que la fuente (todo como texto)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"customerID": [f"{i:04d}-TEST" for i in range(n)]})
    for col, domain in CATEGORICAL_DOMAINS.items():
        df[col] = rng.choice(sorted(domain), size=n)
    df["SeniorCitizen"] = rng.integers(0, 2, n).astype(str)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    df["tenure"] = tenure.astype(str)
    df["MonthlyCharges"] = monthly.astype(str)
    df["TotalCharges"] = [
        " " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly, strict=True)
    ]
    # Señal real: contratos mes a mes y poca antigüedad elevan la fuga.
    logit = -1.0 + 1.5 * (df["Contract"] == "Month-to-month") - 0.03 * tenure
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def cfg(tmp_path, raw_df):
    """Configuración aislada en un directorio temporal, con el CSV ya presente."""
    raw_path = tmp_path / "data" / "raw" / "telco.csv"
    raw_path.parent.mkdir(parents=True)
    raw_df.to_csv(raw_path, index=False)
    from churn.data import sha256_of

    (tmp_path / "config").mkdir()
    config = {
        "data": {
            "url": "https://example.invalid/telco.csv",
            "sha256": sha256_of(raw_path),
            "max_download_bytes": 5_000_000,
            "raw_path": "data/raw/telco.csv",
            "db_path": "data/processed/telco.db",
        },
        "model": {
            "random_state": 0,
            "test_size": 0.2,
            "val_size": 0.2,
            "cv_folds": 3,
            "artifacts_dir": "models",
            "reports_dir": "reports",
        },
        "economics": {"horizon_months": 12, "retention_success_rate": 0.3, "cost_per_contact": 25},
    }
    path = tmp_path / "config" / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_config(path, root=tmp_path)
