# Retención de clientes telecom: a quién llamar y por qué es rentable

🇬🇧 [Read in English](README.md)

Las empresas de telecomunicaciones pierden cerca de una cuarta parte de sus
clientes al año, y los equipos de retención no pueden llamar a todos. Este
proyecto **predice qué clientes están por irse, explica por qué y decide a
quién contactar maximizando el beneficio esperado**, no el accuracy.

![Lista de acción](docs/demo_lista.png)

## Resultados (conjunto de prueba, 1.409 clientes)

| Estrategia | Clientes contactados | Desertores alcanzados | Beneficio de la campaña* |
|---|---:|---:|---:|
| Sin campaña | 0 | 0 | 0 |
| Contactar a todos | 1.409 | 374 / 374 | 62.749 |
| Modelo con umbral 0,5 por defecto | 285 | 193 / 374 | 45.721 |
| **Modelo con umbral óptimo por beneficio (0,09)** | **866** | **355 / 374** | **72.536** |
| Oráculo perfecto (techo teórico) | 374 | 374 / 374 | 88.624 |

\*Supuestos (editables en `config/config.yaml` y en la app): costo por contacto 25,
30 % de éxito de retención, 12 meses de ingreso conservado. Moneda del dataset.

- **+16 % de beneficio frente a contactar a todos y +59 % frente al umbral 0,5**;
  captura el 82 % del máximo teórico.
- ROC-AUC 0,846 · PR-AUC 0,660 · Brier 0,136 (XGBoost, elegido frente a
  regresión logística por PR-AUC en validación cruzada de 5 particiones: 0,670 vs 0,667).
- **El modelo vale más cuando contactar es caro.** Con un costo de 100 por
  contacto, llamar a todos pierde 42.926 y el modelo aún gana 23.925
  (`reports/sensitivity.csv`).
- Principales causas de fuga (SHAP): contrato mes a mes, poca antigüedad,
  cargo mensual alto, sin seguridad en línea y servicio de fibra óptica. Los
  clientes mes a mes se van en un 42,7 %, frente al 2,8 % con contrato de dos años.

![Modelo](docs/demo_modelo.png)

## Cómo funciona

```
CSV público de IBM ──HTTPS + SHA-256──▶ validación de esquema ──▶ SQLite (solo lectura para análisis)
                                                                   │
              división estratificada 60/20/20 ◀────────────────────┘
                 │ entrenamiento: CV de 5 particiones elige LogReg vs XGBoost (PR-AUC)
                 │ validación: umbral que maximiza el beneficio de la campaña
                 │ prueba: se usa una sola vez, reporte final
                 ▼
       model.joblib + manifiesto (SHA-256) ──▶ razones SHAP ──▶ lista de acción / Streamlit
```

Decisiones clave:

- **Beneficio, no accuracy.** Cada contacto cuesta; cada desertor retenido
  devuelve `MonthlyCharges × horizonte × tasa de éxito`. El umbral se elige en
  validación, nunca en prueba.
- **Sin reponderar clases.** El desbalance lo resuelve el umbral económico, así
  las probabilidades quedan calibradas y el valor esperado por cliente
  (`p × éxito × valor − costo`) tiene sentido.
- **SQLite en lugar de Firebase/Supabase.** 7.043 filas, análisis por lotes y un
  solo escritor: una base en archivo, incluida en Python, gratuita, sin
  credenciales ni red y sin superficie de ataque adicional. Una base alojada
  (por ejemplo el plan gratuito de Postgres en Supabase) solo se justifica si la
  app debe guardar estado multiusuario, como el resultado de cada llamada.

## Ejecutar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
churn train                                  # descarga y verifica datos, entrena, escribe reports/
churn score mis_clientes.csv lista.csv       # califica cualquier CSV con el mismo esquema
streamlit run app/streamlit_app.py           # demo interactiva
pytest                                       # 40 pruebas, ~97 % de cobertura
```

## Datos, librerías y licencias

| Componente | Uso | Licencia |
|---|---|---|
| [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) | Dataset (CSV, UTF-8) | Apache-2.0 |
| pandas, NumPy, scikit-learn | Datos y ML | BSD-3-Clause |
| XGBoost | Gradient boosting | Apache-2.0 |
| SHAP | Explicaciones | MIT |
| Streamlit | Demo | Apache-2.0 |
| Matplotlib | Gráficas | Matplotlib (tipo PSF, compatible con BSD) |
| PyYAML, joblib | Configuración, modelos | MIT, BSD-3-Clause |
| SQLite | Almacenamiento | Dominio público |
| Ruff, pytest, Bandit, pip-audit | Calidad y seguridad | MIT, MIT, Apache-2.0, Apache-2.0 |

Ficha completa de los datos en `data/README.md`; controles de seguridad en [SECURITY.md](SECURITY.md).

## Limitaciones (con honestidad)

- Es una sola foto en el tiempo: no hay validación temporal, así que el
  desempeño real sobre los clientes del próximo mes es desconocido.
- Los supuestos de negocio son ilustrativos; la tabla de sensibilidad muestra
  cómo cambian las conclusiones.
- El efecto de la retención se supone, no se mide. En producción debe validarse
  con una prueba A/B (grupo contactado vs. grupo de control).
- Datos ficticios de IBM; un despliegue real requiere datos propios y revisión
  de privacidad (Ley 1581 de 2012).

## Próximos pasos

- Modelado de uplift (a quién le cambia la decisión el contacto, no solo quién se va).
- Curva de calibración y seguimiento de experimentos con MLflow.
- Publicar la demo en Streamlit Community Cloud.

## Autor

Luis Carlos Hernández Urrego · [LinkedIn](https://www.linkedin.com/in/luis-carlos-hernandez-urrego/) · [GitHub](https://github.com/luisk217)

Nace de tres años en operación de clientes de telecomunicaciones (Tigo,
Emtelco), donde decidir a quién llamar antes de que cancele era el trabajo diario.
