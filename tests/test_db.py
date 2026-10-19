import sqlite3

import pytest

from churn.data import clean
from churn.db import churn_rate_by, connect_read_only, read_customers, write_customers


@pytest.fixture
def db(tmp_path, raw_df):
    path = tmp_path / "telco.db"
    write_customers(clean(raw_df), path)
    return path


def test_roundtrip(db, raw_df):
    assert len(read_customers(db)) == len(raw_df)


def test_read_only_connection_blocks_writes(db):
    conn = connect_read_only(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM customers")
    conn.close()


def test_churn_rate_by_valid_column(db):
    result = churn_rate_by(db, "Contract")
    assert set(result["segment"]) == {"Month-to-month", "One year", "Two year"}
    assert result["churn_rate"].between(0, 1).all()


@pytest.mark.parametrize("column", ['Contract"; DROP TABLE customers; --', "Churn", "nope"])
def test_churn_rate_by_rejects_untrusted_column(db, column):
    with pytest.raises(ValueError, match="no permitida"):
        churn_rate_by(db, column)
    assert len(read_customers(db)) > 0
