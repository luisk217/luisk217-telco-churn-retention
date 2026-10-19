import io
from dataclasses import replace

import pytest

from churn.data import (
    DataValidationError,
    IntegrityError,
    clean,
    download_dataset,
    validate_schema,
)


def test_clean_fills_blank_total_charges_for_new_customers(raw_df):
    out = clean(raw_df)
    new = out["tenure"] == 0
    assert new.any()
    assert (out.loc[new, "TotalCharges"] == 0).all()
    assert set(out["Churn"].unique()) <= {0, 1}


def test_rejects_unknown_category(raw_df):
    raw_df.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(DataValidationError, match="Contract"):
        clean(raw_df)


def test_rejects_duplicate_ids(raw_df):
    raw_df.loc[1, "customerID"] = raw_df.loc[0, "customerID"]
    with pytest.raises(DataValidationError, match="duplicados"):
        clean(raw_df)


def test_rejects_negative_charges(raw_df):
    raw_df.loc[0, "MonthlyCharges"] = "-5"
    with pytest.raises(DataValidationError, match="MonthlyCharges"):
        clean(raw_df)


def test_rejects_missing_column(raw_df):
    with pytest.raises(DataValidationError, match="Faltan"):
        validate_schema(raw_df.drop(columns=["tenure"]))


def test_scoring_data_without_target(raw_df):
    out = clean(raw_df.drop(columns=["Churn"]), require_target=False)
    assert "Churn" not in out


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_download_rejects_tampered_file(cfg, monkeypatch):
    cfg.data.raw_path.unlink()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse(b"customerID\nmalicioso\n")
    )
    with pytest.raises(IntegrityError, match="SHA-256"):
        download_dataset(cfg.data)
    assert not cfg.data.raw_path.exists()
    assert not list(cfg.data.raw_path.parent.glob("*.part"))


def test_download_enforces_size_limit(cfg, monkeypatch):
    cfg.data.raw_path.unlink()
    small = replace(cfg.data, max_download_bytes=10)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(b"x" * 100))
    with pytest.raises(IntegrityError, match="tamaño"):
        download_dataset(small)


def test_download_accepts_matching_hash(cfg, monkeypatch):
    content = cfg.data.raw_path.read_bytes()
    cfg.data.raw_path.unlink()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(content))
    assert download_dataset(cfg.data).read_bytes() == content
