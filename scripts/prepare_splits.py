#!/usr/bin/env python3
"""Create deterministic development and heldout splits from official training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prior.data_preparation import materialize_evaluation_splits, split_config_values
from prior.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--max-rows", type=int, help="Testing only: stop after N input rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed, development_fraction, heldout_fraction = split_config_values(load_json(args.config))
    summary = materialize_evaluation_splits(
        args.train_jsonl,
        args.output_dir,
        seed=seed,
        development_fraction=development_fraction,
        heldout_fraction=heldout_fraction,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
