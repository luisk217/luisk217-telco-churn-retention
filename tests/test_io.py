import pandas as pd
import pytest

from churn.io import MAX_UPLOAD_BYTES, neutralize_formula, read_uploaded_csv, safe_csv_export


@pytest.mark.parametrize("value", ['=HYPERLINK("x")', "+1", "-2+3", "@SUM(A1)"])
def test_neutralizes_formulas(value):
    assert neutralize_formula(value) == "'" + value


def test_leaves_normal_values():
    assert neutralize_formula("7590-VHVEG") == "7590-VHVEG"
    assert neutralize_formula(-3.5) == -3.5


def test_safe_csv_export():
    csv = safe_csv_export(pd.DataFrame({"a": ["=cmd"], "b": [1]})).decode()
    assert "'=cmd" in csv


def test_upload_size_limit():
    with pytest.raises(ValueError, match="supera"):
        read_uploaded_csv(b"x" * (MAX_UPLOAD_BYTES + 1))


def test_upload_requires_utf8():
    with pytest.raises(ValueError, match="UTF-8"):
        read_uploaded_csv("ñ".encode("latin-1"))
