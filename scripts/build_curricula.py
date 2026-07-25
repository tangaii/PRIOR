#!/usr/bin/env python3
"""Materialize deterministic PRIOR training curricula."""

from __future__ import annotations

import argparse
from pathlib import Path

from prior.expert_specialization import build_curricula, curriculum_specs
from prior.io import iter_jsonl, write_json_atomic, write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--curricula",
        nargs="+",
        default=list(curriculum_specs()),
        choices=list(curriculum_specs()),
    )
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded: set[str] = set()
    for path in args.exclude_jsonl:
        excluded.update(str(row["sample_id"]) for row in iter_jsonl(path))
    available = curriculum_specs()
    selected = [available[name] for name in args.curricula]
    curricula, summary = build_curricula(
        iter_jsonl(args.train_jsonl),
        selected,
        excluded_sample_ids=excluded,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in curricula.items():
        write_jsonl_atomic(args.output_dir / f"{name}.jsonl", rows)
    write_json_atomic(args.output_dir / "curriculum_summary.json", summary)


if __name__ == "__main__":
    main()
