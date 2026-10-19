"""Guardado y carga de modelos con verificación de integridad.

joblib usa pickle, y cargar un pickle ejecuta código: solo se cargan
modelos cuyo SHA-256 coincide con el manifiesto escrito al entrenar.
Nunca cargues un .joblib recibido de una fuente no confiable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from churn.data import sha256_of

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"


class ArtifactIntegrityError(RuntimeError):
    """El modelo en disco no coincide con su manifiesto."""


def save_model(model: Any, metadata: dict[str, Any], artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_FILE
    joblib.dump(model, model_path)
    manifest = {**metadata, "model_file": MODEL_FILE, "model_sha256": sha256_of(model_path)}
    (artifacts_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return model_path


def load_manifest(artifacts_dir: Path) -> dict[str, Any]:
    return json.loads((artifacts_dir / MANIFEST_FILE).read_text(encoding="utf-8"))


def load_model(artifacts_dir: Path) -> tuple[Any, dict[str, Any]]:
    manifest = load_manifest(artifacts_dir)
    model_path = artifacts_dir / MODEL_FILE
    if sha256_of(model_path) != manifest.get("model_sha256"):
        raise ArtifactIntegrityError("El hash del modelo no coincide con el manifiesto.")
    # Seguro solo porque el hash se verificó contra el manifiesto propio.
    return joblib.load(model_path), manifest  # nosec B301
