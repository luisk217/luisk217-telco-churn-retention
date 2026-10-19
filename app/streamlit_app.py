"""Demo interactiva: lista de retención priorizada por valor esperado.

Ejecutar con:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import streamlit as st

from churn.artifacts import MANIFEST_FILE, load_model
from churn.config import load_config
from churn.data import CATEGORICAL_COLUMNS, DataValidationError, clean
from churn.db import churn_rate_by, read_customers
from churn.economics import customer_value, expected_value
from churn.explain import shap_values, top_reasons
from churn.io import read_uploaded_csv, safe_csv_export
from churn.pipeline import score, train

st.set_page_config(page_title="Retención de clientes telecom", layout="wide")
CFG = load_config()
TOP_N_REASONS = 200


@st.cache_resource(show_spinner="Entrenando el modelo por primera vez (unos segundos)...")
def ensure_model() -> dict:
    """En un despliegue limpio no hay modelo: se entrena desde la fuente verificada."""
    manifest = CFG.model.artifacts_dir / MANIFEST_FILE
    if not manifest.exists() or not CFG.data.db_path.exists():
        train(CFG)
    return json.loads((CFG.model.reports_dir / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Calculando probabilidades...")
def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    return score(df, CFG, with_reasons=False)


metrics = ensure_model()

st.title("Predicción de fuga de clientes con impacto económico")
st.caption(
    "Dataset público IBM Telco Customer Churn (Apache-2.0). "
    "Los montos están en la moneda del dataset; los supuestos se ajustan en la barra lateral."
)

with st.sidebar:
    st.header("Supuestos de negocio")
    econ = replace(
        CFG.economics,
        cost_per_contact=st.slider(
            "Costo por contacto", 0.0, 200.0, CFG.economics.cost_per_contact
        ),
        retention_success_rate=st.slider(
            "Tasa de éxito de retención", 0.05, 1.0, CFG.economics.retention_success_rate
        ),
        horizon_months=st.slider("Horizonte (meses)", 1, 36, CFG.economics.horizon_months),
    )
    st.caption("Se contacta a un cliente si el valor esperado del contacto es positivo.")
    st.header("Datos")
    uploaded = st.file_uploader("CSV propio con el mismo esquema (opcional)", type=["csv"])

if uploaded is not None:
    try:
        customers = clean(read_uploaded_csv(uploaded.getvalue()), require_target=False)
    except (ValueError, DataValidationError) as exc:
        # El mensaje de validación no incluye contenido del archivo, solo la causa.
        st.error(f"El archivo no es válido: {exc}")
        st.stop()
else:
    customers = read_customers(CFG.data.db_path)

scored = score_customers(customers)
value = customer_value(
    customers.set_index("customerID").loc[scored["customerID"], "MonthlyCharges"], econ
)
scored["revenue_at_risk"] = value.round(2)
scored["expected_value_of_contact"] = expected_value(
    scored["churn_probability"].to_numpy(), value, econ
).round(2)
scored["contact"] = scored["expected_value_of_contact"] > 0
scored = scored.sort_values("expected_value_of_contact", ascending=False).reset_index(drop=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", f"{len(scored):,}")
c2.metric("A contactar", f"{int(scored['contact'].sum()):,}")
c3.metric(
    "Valor esperado de la campaña",
    f"{scored.loc[scored['contact'], 'expected_value_of_contact'].sum():,.0f}",
)
c4.metric("ROC-AUC en prueba", metrics["test_classification"]["roc_auc"])

tab_list, tab_model, tab_segments = st.tabs(["Lista de acción", "Modelo", "Segmentos"])

with tab_list:
    top = scored.head(TOP_N_REASONS).copy()
    indexed = customers.set_index("customerID").loc[top["customerID"]].reset_index()
    background = customers.sample(min(100, len(customers)), random_state=CFG.model.random_state)
    model, _ = load_model(CFG.model.artifacts_dir)
    top["top_reasons"] = top_reasons(shap_values(model, indexed, background)).to_numpy()
    st.subheader(f"Los {len(top)} clientes con mayor valor esperado de contacto")
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar lista completa (CSV)",
        data=safe_csv_export(scored),
        file_name="lista_retencion.csv",
        mime="text/csv",
    )

with tab_model:
    camp = metrics["test_campaign"]
    st.markdown(
        f"Modelo elegido por PR-AUC en validación cruzada: **{metrics['model']}**. "
        f"Con el umbral óptimo **{metrics['threshold']}**, en el conjunto de prueba la campaña "
        f"genera **{camp['profit_model']:,.0f}**, frente a "
        f"**{camp['profit_contact_everyone']:,.0f}** si se contacta a todos y "
        f"**{camp['profit_perfect_oracle']:,.0f}** con un oráculo perfecto."
    )
    left, right = st.columns(2)
    left.image(str(CFG.model.reports_dir / "profit_curve.png"))
    right.image(str(CFG.model.reports_dir / "shap_importance.png"))
    st.subheader("Sensibilidad a los supuestos")
    st.dataframe(pd.read_csv(CFG.model.reports_dir / "sensitivity.csv"), hide_index=True)
    st.json(metrics["test_classification"])

with tab_segments:
    column = st.selectbox(
        "Segmentar por", CATEGORICAL_COLUMNS, index=CATEGORICAL_COLUMNS.index("Contract")
    )
    st.dataframe(churn_rate_by(CFG.data.db_path, column), hide_index=True)
