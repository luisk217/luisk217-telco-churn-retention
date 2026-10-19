"""Persistencia en SQLite: escritura controlada y lectura en modo solo lectura."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

TABLE_CUSTOMERS = "customers"


def write_customers(df: pd.DataFrame, db_path: Path) -> None:
    """Reemplaza la tabla de clientes de forma atómica dentro de una transacción."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        df.to_sql(TABLE_CUSTOMERS, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_customer_id ON {TABLE_CUSTOMERS}(customerID)"
        )


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Abre la base en modo `ro`: ninguna consulta puede modificar los datos."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def read_customers(db_path: Path) -> pd.DataFrame:
    with closing(connect_read_only(db_path)) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_CUSTOMERS}", conn)  # noqa: S608  # nosec B608


def churn_rate_by(db_path: Path, column: str) -> pd.DataFrame:
    """Tasa de fuga agrupada por una columna permitida.

    Los nombres de columna no pueden parametrizarse en SQL, así que se
    validan contra la lista blanca de columnas reales de la tabla.
    """
    with closing(connect_read_only(db_path)) as conn:
        allowed = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_CUSTOMERS})")}
        if column not in allowed or column in {"customerID", "Churn"}:
            raise ValueError(f"Columna no permitida: {column!r}")
        query = (
            f'SELECT "{column}" AS segment, COUNT(*) AS customers, '  # noqa: S608  # nosec B608
            f"ROUND(AVG(Churn), 4) AS churn_rate "
            f'FROM {TABLE_CUSTOMERS} GROUP BY "{column}" ORDER BY churn_rate DESC'
        )
        return pd.read_sql_query(query, conn)
