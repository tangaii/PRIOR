#!/usr/bin/env python3
"""Validate a PRIOR or challenge-format submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prior.validation import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_submission(
        args.submission,
        test_path=args.test_jsonl,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
