"""Deterministic public-data partitioning for PRIOR retraining."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import iter_jsonl, sha256_file, write_json_atomic
from .schema import source_prefix


def split_assignment(
    sample_id: str,
    *,
    seed: int,
    development_fraction: float,
    heldout_fraction: float,
) -> str:
    """Assign one sample using a stable hash independent of input order."""

    if development_fraction < 0 or heldout_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if development_fraction + heldout_fraction >= 1:
        raise ValueError("development and heldout fractions must sum to less than one")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < development_fraction:
        return "development"
    if value < development_fraction + heldout_fraction:
        return "heldout"
    return "training"


def materialize_evaluation_splits(
    train_jsonl: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    development_fraction: float,
    heldout_fraction: float,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Write full development/heldout rows and a compact split summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        name: output / f"{name}.jsonl" for name in ("development", "heldout")
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in destinations.items()
    }
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("training", "development", "heldout")
    }
    handles = {}
    try:
        handles = {
            name: path.open("w", encoding="utf-8") for name, path in temporary.items()
        }
        for row_number, row in enumerate(iter_jsonl(train_jsonl), start=1):
            if max_rows is not None and row_number > max_rows:
                break
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"missing sample_id at training row {row_number}")
            split = split_assignment(
                sample_id,
                seed=seed,
                development_fraction=development_fraction,
                heldout_fraction=heldout_fraction,
            )
            counts[split] += 1
            source_counts[split][source_prefix(sample_id)] += 1
            if split in handles:
                handles[split].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        for name, path in destinations.items():
            os.replace(temporary[name], path)
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    summary: dict[str, Any] = {
        "protocol": "public_retraining_v1",
        "seed": seed,
        "development_fraction": development_fraction,
        "heldout_fraction": heldout_fraction,
        "rows": dict(counts),
        "sources": {name: dict(sorted(values.items())) for name, values in source_counts.items()},
        "files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in destinations.items()
        },
    }
    write_json_atomic(output / "split_summary.json", summary)
    return summary


def split_config_values(config: Mapping[str, Any]) -> tuple[int, float, float]:
    """Read and validate the three public split parameters."""

    seed = int(config["seed"])
    development_fraction = float(config["development_fraction"])
    heldout_fraction = float(config["heldout_fraction"])
    split_assignment(
        "validation-probe",
        seed=seed,
        development_fraction=development_fraction,
        heldout_fraction=heldout_fraction,
    )
    return seed, development_fraction, heldout_fraction
