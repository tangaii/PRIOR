#!/usr/bin/env python3
"""Train one shared, specialist, or anchor-repair stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from prior.io import iter_jsonl, load_json
from prior.training import train_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.stage not in config["stages"]:
        raise KeyError(f"unknown training stage: {args.stage}")
    stage = dict(config["stages"][args.stage])
    stage["warmup_ratio"] = config["warmup_ratio"]
    stage["gradient_clip"] = config["gradient_clip"]
    rows = list(iter_jsonl(args.curriculum))
    seed = int(stage.get("seed", config["seed"]))
    train_stage(
        rows,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        stage=stage,
        seed=seed,
        source_checkpoint=args.source_checkpoint,
    )


if __name__ == "__main__":
    main()
