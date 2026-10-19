# Data card / Ficha de datos

| Field | Value |
|---|---|
| Name | Telco Customer Churn (IBM sample data) |
| Source | [`IBM/telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d) → `data/Telco-Customer-Churn.csv` |
| License | Apache License 2.0 (license of the source repository) |
| Format | CSV, UTF-8, comma-separated, header row (RFC 4180) |
| Size | 7,043 rows × 21 columns, ~955 KB |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Personal data | None. Fictional customers created by IBM; `customerID` is synthetic. |
| Target | `Churn` (Yes/No), 26.5 % positive |

## How it is obtained

`churn train` downloads the file over HTTPS, enforces a 5 MB limit, and only
keeps it if the SHA-256 matches the value above (`config/config.yaml`). Raw and
processed files are **not committed**; they are rebuilt on demand.

## Why this source and not Kaggle

The same dataset is mirrored on Kaggle (`blastchar/telco-customer-churn`), but
that page does not state a clear license and requires an account to download.
The IBM repository is public, requires no credentials, and is Apache-2.0.

## Known quirks

- `TotalCharges` is blank for the 11 customers with `tenure == 0` (they have not
  been billed yet). The pipeline sets them to `0`, which is the correct value.
- Monetary columns have no currency stated. Results are reported in "dataset
  units" and the business assumptions in `config/config.yaml` must use the same unit.
- It is a single snapshot: there is no time dimension, so the model cannot be
  validated on a later period. See the README limitations.

## Storage

The cleaned table is stored in SQLite (`data/processed/telco.db`, table
`customers`, unique index on `customerID`). The app and analysis code open it
in read-only mode (`?mode=ro`).
