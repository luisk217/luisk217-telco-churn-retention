"""Interfaz de línea de comandos: `churn train` y `churn score`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from churn.config import DEFAULT_CONFIG_PATH, load_config
from churn.data import clean, load_raw
from churn.io import safe_csv_export
from churn.pipeline import score, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Descarga datos, entrena y genera reportes.")
    p_score = sub.add_parser("score", help="Genera la lista de acción para un CSV de clientes.")
    p_score.add_argument("input", type=Path)
    p_score.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.command == "train":
        print(json.dumps(train(cfg), indent=2))
    else:
        df = clean(load_raw(args.input), require_target=False)
        args.output.write_bytes(safe_csv_export(score(df, cfg)))
        print(f"Lista de acción escrita en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
