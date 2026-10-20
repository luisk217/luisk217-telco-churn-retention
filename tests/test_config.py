from pathlib import Path

import pytest
import yaml

from churn.config import ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _write(tmp_path, mutate):
    raw = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_is_valid():
    cfg = load_config(REPO_CONFIG)
    assert cfg.data.url.startswith("https://")
    assert cfg.data.raw_path.is_relative_to(REPO_CONFIG.parents[1])


def test_rejects_plain_http(tmp_path):
    path = _write(tmp_path, lambda r: r["data"].update(url="http://example.com/x.csv"))
    with pytest.raises(ConfigError, match="HTTPS"):
        load_config(path, root=tmp_path)


def test_rejects_path_traversal(tmp_path):
    path = _write(tmp_path, lambda r: r["data"].update(raw_path="../../etc/passwd"))
    with pytest.raises(ConfigError, match="sale del directorio"):
        load_config(path, root=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("economics", "retention_success_rate", 1.5),
        ("economics", "cost_per_contact", -1),
        ("model", "test_size", 0.9),
        ("data", "sha256", "abc"),
    ],
)
def test_rejects_invalid_values(tmp_path, section, key, value):
    path = _write(tmp_path, lambda r: r[section].update({key: value}))
    with pytest.raises(ConfigError):
        load_config(path, root=tmp_path)


def test_missing_section(tmp_path):
    path = _write(tmp_path, lambda r: r.pop("economics"))
    with pytest.raises(ConfigError, match="Falta"):
        load_config(path, root=tmp_path)
