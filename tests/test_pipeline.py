import json

import pytest

from churn.artifacts import MODEL_FILE, ArtifactIntegrityError, load_model
from churn.cli import main
from churn.data import clean
from churn.pipeline import score, train


@pytest.fixture
def trained(cfg):
    return train(cfg)


def test_train_writes_reports_and_model(cfg, trained):
    assert trained["model"] in {"logistic_regression", "xgboost"}
    assert 0 < trained["threshold"] < 1
    assert trained["test_classification"]["roc_auc"] > 0.6
    for name in ("metrics.json", "profit_curve.png", "shap_importance.csv", "sensitivity.csv"):
        assert (cfg.model.reports_dir / name).exists()
    saved = json.loads((cfg.model.reports_dir / "metrics.json").read_text())
    assert saved["threshold"] == trained["threshold"]


def test_score_ranks_by_expected_value(cfg, trained, raw_df):
    df = clean(raw_df.drop(columns=["Churn"]).head(30), require_target=False)
    out = score(df, cfg)
    assert len(out) == 30
    assert out["expected_value_of_contact"].is_monotonic_decreasing
    assert out["churn_probability"].between(0, 1).all()
    assert out["top_reasons"].str.len().gt(0).all()


def test_tampered_model_is_rejected(cfg, trained):
    with open(cfg.model.artifacts_dir / MODEL_FILE, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        load_model(cfg.model.artifacts_dir)


def test_cli_score(cfg, trained, raw_df, tmp_path):
    config_path = tmp_path / "config" / "config.yaml"
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    raw_df.drop(columns=["Churn"]).head(10).to_csv(src, index=False)
    assert main(["--config", str(config_path), "score", str(src), str(dst)]) == 0
    assert dst.read_text().startswith("customerID,churn_probability")
