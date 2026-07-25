#!/usr/bin/env python3
"""Run the complete public PRIOR neural-training sequence."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from prior.io import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curricula-dir", type=Path, default=Path("data/curricula"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/prior"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--devices", default="0", help="Comma-separated CUDA ids, or 'cpu'.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stage_command(
    stage: str,
    *,
    curricula_dir: Path,
    model_dir: Path,
    output_dir: Path,
    config: Path,
    source_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--config",
        str(config),
        "--stage",
        stage,
        "--curriculum",
        str(curricula_dir / f"{stage}.jsonl"),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir / stage),
    ]
    if source_checkpoint is not None:
        command.extend(["--source-checkpoint", str(source_checkpoint)])
    return command


def stage_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def main() -> None:
    args = parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if args.devices == "cpu":
        devices = ["cpu"]
    if not devices:
        raise ValueError("at least one device is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        stage: args.output_dir / stage / "checkpoint.pt"
        for stage in (
            "shared_initialization",
            "shared_continuation",
            "coverage_expert",
            "precision_expert",
            "anchor_repair",
        )
    }
    sources = {
        "shared_initialization": None,
        "shared_continuation": checkpoints["shared_initialization"],
        "coverage_expert": checkpoints["shared_continuation"],
        "precision_expert": checkpoints["shared_continuation"],
        "anchor_repair": checkpoints["precision_expert"],
    }
    plan = {
        stage: stage_command(
            stage,
            curricula_dir=args.curricula_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            config=args.config,
            source_checkpoint=sources[stage],
        )
        for stage in checkpoints
    }
    if args.dry_run:
        print(json.dumps({stage: shlex.join(command) for stage, command in plan.items()}, indent=2))
        return

    completed: list[str] = []

    def should_run(stage: str) -> bool:
        if checkpoints[stage].exists() and not args.force:
            completed.append(f"{stage}:reused")
            return False
        curriculum = args.curricula_dir / f"{stage}.jsonl"
        if not curriculum.is_file():
            raise FileNotFoundError(f"missing curriculum: {curriculum}")
        source = sources[stage]
        if source is not None and not source.is_file():
            raise FileNotFoundError(f"missing source checkpoint for {stage}: {source}")
        return True

    for stage in ("shared_initialization", "shared_continuation"):
        if should_run(stage):
            subprocess.run(plan[stage], check=True, env=stage_environment(devices[0]))
            completed.append(stage)

    specialists = [stage for stage in ("coverage_expert", "precision_expert") if should_run(stage)]
    if len(devices) >= 2 and len(specialists) == 2:
        processes = {
            stage: subprocess.Popen(plan[stage], env=stage_environment(devices[index]))
            for index, stage in enumerate(specialists)
        }
        failed = []
        for stage, process in processes.items():
            return_code = process.wait()
            if return_code:
                failed.append((stage, return_code))
            else:
                completed.append(stage)
        if failed:
            raise RuntimeError(f"specialist training failed: {failed}")
    else:
        for stage in specialists:
            subprocess.run(plan[stage], check=True, env=stage_environment(devices[0]))
            completed.append(stage)

    if should_run("anchor_repair"):
        subprocess.run(plan["anchor_repair"], check=True, env=stage_environment(devices[0]))
        completed.append("anchor_repair")

    summary: dict[str, Any] = {
        "protocol": "public_retraining_v1",
        "devices": devices,
        "completed": completed,
        "checkpoints": {stage: str(path) for stage, path in checkpoints.items()},
    }
    write_json_atomic(args.output_dir / "training_pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
