#!/usr/bin/env python3
"""Run a frozen PRIOR tagger and emit row-aligned predictions."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from prior.anchor_refinement import refine_anchor_row
from prior.io import iter_jsonl, load_json, write_jsonl_atomic
from prior.risk_aware_span_tagging import load_tagger, load_tokenizer, predict_scored_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=("scored", "expert", "anchor"), required=True)
    parser.add_argument("--anchor-config", type=Path, default=Path("configs/anchor.json"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def batches(path: Path, size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    import torch

    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_dir)
    model = load_tagger(args.model_dir, device=device, checkpoint=args.checkpoint)
    anchor = load_json(args.anchor_config) if args.mode == "anchor" else None

    def output_rows() -> Iterator[dict[str, Any]]:
        for rows in batches(args.input_jsonl, args.batch_size):
            predictions = predict_scored_spans(
                rows,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            for row in rows:
                sample_id = str(row["sample_id"])
                spans = predictions[sample_id]
                if args.mode == "anchor":
                    assert anchor is not None
                    yield refine_anchor_row(
                        row,
                        spans,
                        thresholds=anchor["thresholds"],
                        trim_characters=anchor["trim_characters"],
                    )
                elif args.mode == "scored":
                    yield {"sample_id": sample_id, "pred_spans": spans}
                else:
                    yield {
                        "sample_id": sample_id,
                        "pred_spans": [
                            {"start": span["start"], "end": span["end"], "label": span["label"]}
                            for span in spans
                        ],
                    }

    write_jsonl_atomic(args.output_jsonl, output_rows(), compact=True)


if __name__ == "__main__":
    main()
