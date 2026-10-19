# Security / Seguridad

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's
[private vulnerability reporting](../../security/advisories/new) for this repository.

## Controls in this project

| Risk | Control | Where |
|---|---|---|
| Tampered or swapped dataset | HTTPS only, size limit, SHA-256 check before the file is kept; atomic write | `src/churn/data.py`, `src/churn/config.py` |
| Malicious YAML config | `yaml.safe_load` + typed validation; paths cannot leave the project root | `src/churn/config.py` |
| SQL injection | No user value is concatenated into SQL; column names checked against an allow-list from `PRAGMA table_info`; read-only connections | `src/churn/db.py` |
| Untrusted model files (pickle) | Models only load if their SHA-256 matches the manifest written at training | `src/churn/artifacts.py` |
| Malformed uploads | 5 MB limit, UTF-8 only, no type inference, strict schema and category validation | `src/churn/io.py`, `src/churn/data.py`, `.streamlit/config.toml` |
| CSV/formula injection in exports | Text cells starting with `= + - @` are neutralized | `src/churn/io.py` |
| Vulnerable dependencies | `pip-audit` in CI (weekly too) + Dependabot | `.github/` |
| Insecure code patterns | Ruff `S` rules + Bandit in CI and pre-commit | `pyproject.toml`, `.pre-commit-config.yaml` |
| Over-privileged CI | `permissions: contents: read`, no persisted credentials | `.github/workflows/ci.yml` |
| Secrets in the repo | None are needed; `.env` and Streamlit secrets are git-ignored; `detect-private-key` hook | `.gitignore` |
| Personal data | Only fictional public data is used. Do not load real customer data into the public demo. | `data/README.md` |
