#!/usr/bin/env python3
"""Evaluate expert predictions by source and emit a router-selection audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prior.io import iter_jsonl, write_json_atomic
from prior.policy_evaluation import (
    build_gold_spans,
    compute_sample_metrics,
    read_predictions,
    record_text,
)
from prior.schema import source_prefix
from prior.whole_output_routing import choose_source_routes


EXPERTS = ("anchor", "coverage", "precision")
SPLITS = {"development": "development", "heldout": "fresh"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-output", type=Path)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    parser.add_argument("--critical-threshold", type=float, default=1.0)
    parser.add_argument("--fp-char-rate-threshold", type=float, default=0.05)
    parser.add_argument("--tolerance", type=int, default=2)
    return parser.parse_args()


def new_totals() -> dict[str, float]:
    return {
        "rows": 0,
        "passes": 0,
        "weighted_sum": 0.0,
        "critical_sum": 0.0,
        "fp_sum": 0.0,
    }


def finalized(values: dict[str, float]) -> dict[str, Any]:
    rows = int(values["rows"])
    return {
        "rows": rows,
        "passes": int(values["passes"]),
        "pcr": values["passes"] / rows if rows else 0.0,
        "weighted_coverage": values["weighted_sum"] / rows if rows else 0.0,
        "critical_coverage": values["critical_sum"] / rows if rows else 0.0,
        "fp_char_rate": values["fp_sum"] / rows if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    totals: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: {
            expert: {"development": new_totals(), "fresh": new_totals(), "all": new_totals()}
            for expert in EXPERTS
        }
    )
    split_rows: dict[str, int] = {}
    for split_file, audit_split in SPLITS.items():
        data_path = args.splits_dir / f"{split_file}.jsonl"
        predictions = {
            expert: read_predictions(args.predictions_dir / expert / f"{split_file}.jsonl")
            for expert in EXPERTS
        }
        seen: set[str] = set()
        for row in iter_jsonl(data_path):
            sample_id = str(row["sample_id"])
            source = source_prefix(sample_id)
            seen.add(sample_id)
            gold = build_gold_spans(row)
            for expert in EXPERTS:
                if sample_id not in predictions[expert]:
                    raise KeyError(f"missing {expert} prediction for {sample_id} in {split_file}")
                detail = compute_sample_metrics(
                    sample_id=sample_id,
                    dataset=source,
                    text=record_text(row),
                    gt_spans=gold,
                    pred_spans=predictions[expert][sample_id]["pred_spans"],
                    coverage_threshold=args.coverage_threshold,
                    critical_threshold=args.critical_threshold,
                    fp_char_rate_threshold=args.fp_char_rate_threshold,
                    tolerance=args.tolerance,
                )
                for target in (audit_split, "all"):
                    values = totals[source][expert][target]
                    values["rows"] += 1
                    values["passes"] += int(detail["policy_pass"])
                    values["weighted_sum"] += float(detail["weighted_pii_coverage"])
                    values["critical_sum"] += float(detail["critical_coverage"])
                    values["fp_sum"] += float(detail["fp_char_rate"])
        for expert, rows in predictions.items():
            extras = set(rows) - seen
            if extras:
                raise ValueError(f"{expert} has {len(extras)} extra rows in {split_file}")
        split_rows[split_file] = len(seen)

    audit = {
        source: {
            expert: {split: finalized(values) for split, values in split_values.items()}
            for expert, split_values in experts.items()
        }
        for source, experts in sorted(totals.items())
    }
    routes = choose_source_routes(audit)
    payload = {
        "protocol": "public_retraining_v1",
        "split_rows": split_rows,
        "audit": audit,
        "selected_routes": routes,
    }
    write_json_atomic(args.output, payload)
    if args.router_output:
        write_json_atomic(
            args.router_output,
            {
                "default_expert": "anchor",
                "source_routes": routes,
                "source_parser": "prefix_before_first_colon",
                "selection_unit": "complete_prediction_set",
                "span_fusion": False,
                "post_selection_modification": False,
            },
        )
    print(json.dumps({"output": str(args.output), "selected_routes": routes}, indent=2))


if __name__ == "__main__":
    main()
