#!/usr/bin/env python3
"""Freeze a source router from a precomputed local source audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from prior.io import load_json, write_json_atomic
from prior.whole_output_routing import choose_source_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_json(args.audit_json)
    routes = choose_source_routes(audit)
    write_json_atomic(
        args.output_json,
        {
            "default_expert": "anchor",
            "source_routes": routes,
            "source_parser": "prefix_before_first_colon",
            "selection_unit": "complete_prediction_set",
            "span_fusion": False,
            "post_selection_modification": False,
        },
    )


if __name__ == "__main__":
    main()
