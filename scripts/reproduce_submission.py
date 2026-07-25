#!/usr/bin/env python3
"""Route expert outputs, with optional byte-for-byte frozen reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

from prior.io import load_json, sha256_file, verify_file, write_json_atomic
from prior.validation import validate_submission
from prior.whole_output_routing import route_prediction_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/frozen"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/artifacts.json"))
    parser.add_argument("--router", type=Path, default=Path("configs/router.json"))
    parser.add_argument("--output", type=Path, default=Path("reproduction/prior_submission.jsonl"))
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--report", type=Path, default=Path("reproduction/reproduction_report.json"))
    parser.add_argument("--anchor", type=Path, help="Custom anchor prediction JSONL.")
    parser.add_argument("--coverage", type=Path, help="Custom coverage prediction JSONL.")
    parser.add_argument("--precision", type=Path, help="Custom precision prediction JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    router = load_json(args.router)
    custom_paths = {"anchor": args.anchor, "coverage": args.coverage, "precision": args.precision}
    custom_mode = any(path is not None for path in custom_paths.values())
    if custom_mode and not all(path is not None for path in custom_paths.values()):
        raise ValueError("custom routing requires --anchor, --coverage, and --precision")
    manifest = None
    if custom_mode:
        expert_paths = {name: path for name, path in custom_paths.items() if path is not None}
    else:
        manifest = load_json(args.manifest)
        expert_paths = {}
        for name, expected in manifest["inputs"].items():
            path = args.artifact_dir / expected["filename"]
            verify_file(
                path,
                expected_sha256=expected["sha256"],
                expected_size=expected["size_bytes"],
            )
            expert_paths[name] = path
    route_stats = route_prediction_files(
        expert_paths,
        args.output,
        routes=router["source_routes"],
        default_expert=router["default_expert"],
    )
    output_sha256 = sha256_file(args.output)
    output_size = args.output.stat().st_size
    expected_rows = args.expected_rows
    if expected_rows is None and manifest is not None:
        expected_rows = int(manifest["row_count"])
    audit = validate_submission(
        args.output,
        test_path=args.test_jsonl,
        expected_rows=expected_rows,
    )
    output_report = {
        "path": str(args.output),
        "size_bytes": output_size,
        "sha256": output_sha256,
    }
    if manifest is not None:
        output_contract = manifest["output"]
        output_report.update(
            {
                "expected_size_bytes": output_contract["size_bytes"],
                "expected_sha256": output_contract["sha256"],
                "bitwise_identical": (
                    output_sha256 == output_contract["sha256"]
                    and output_size == int(output_contract["size_bytes"])
                ),
            }
        )
    report = {
        "artifact_level_reproduction": not custom_mode,
        "custom_prediction_routing": custom_mode,
        "route_stats": route_stats,
        "output": output_report,
        "validation": audit,
    }
    write_json_atomic(args.report, report)
    if (manifest is not None and not output_report["bitwise_identical"]) or not audit["strict_pass"]:
        raise SystemExit("REPRODUCTION_FAILED")


if __name__ == "__main__":
    main()
