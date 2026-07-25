"""Training engine shared by the initialization and specialist stages."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import write_json_atomic
from .risk_aware_span_tagging import (
    align_training_batch,
    load_tagger,
    load_tokenizer,
    risk_weighted_cross_entropy,
    split_window_batch,
)
from .schema import build_bio_label_maps


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context() -> tuple[int, int, int]:
    """Return world size, global rank, and local rank from torchrun."""

    return (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def _partition_indices(
    size: int,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    rank: int,
) -> list[int]:
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    if world_size > 1 and indices:
        padded_size = math.ceil(len(indices) / world_size) * world_size
        indices.extend(indices[: padded_size - len(indices)])
    return indices[rank::world_size]


def _atomic_torch_save(payload: Any, output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def train_stage(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    stage: Mapping[str, Any],
    seed: int,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one configured stage, optionally under ``torchrun`` DDP.

    The stage produces only ``checkpoint.pt`` and ``training_summary.json``.
    It intentionally avoids retaining per-step checkpoints and predictions.
    """

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    if not rows:
        raise ValueError("training curriculum is empty")
    world_size, rank, local_rank = distributed_context()
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed_everything(seed + rank)
    tokenizer = load_tokenizer(model_dir)
    model = load_tagger(
        model_dir,
        device=device,
        checkpoint=source_checkpoint,
        train_mode=True,
    )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    learning_rate = float(stage["learning_rate"])
    raw_batch_size = int(stage["raw_batch_size"])
    window_chunk_size = int(stage["window_chunk_size"])
    max_length = int(stage["max_length"])
    stride = int(stage["stride"])
    epochs = int(stage.get("epochs", 10_000))
    max_updates = stage.get("optimizer_updates")
    local_rows_per_epoch = math.ceil(len(rows) / world_size)
    updates_per_epoch = math.ceil(local_rows_per_epoch / raw_batch_size)
    total_updates = int(max_updates) if max_updates is not None else epochs * updates_per_epoch
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(stage.get("warmup_ratio", 0.06))),
        num_training_steps=max(1, total_updates),
    )
    label_to_id, _ = build_bio_label_maps()
    autocast_enabled = device.type == "cuda"
    running_loss = 0.0
    updates = 0
    completed_epochs = 0

    for epoch in range(epochs):
        indices = _partition_indices(
            len(rows),
            seed=seed,
            epoch=epoch,
            world_size=world_size,
            rank=rank,
        )
        for batch_start in range(0, len(indices), raw_batch_size):
            if updates >= total_updates:
                break
            records = [rows[index] for index in indices[batch_start : batch_start + raw_batch_size]]
            batch = align_training_batch(
                records,
                tokenizer,
                label_to_id,
                max_length=max_length,
                stride=stride,
            )
            optimizer.zero_grad(set_to_none=True)
            chunk_count = max(
                1,
                math.ceil(int(batch["input_ids"].shape[0]) / window_chunk_size),
            )
            batch_loss = 0.0
            for chunk in split_window_batch(batch, window_chunk_size):
                labels = chunk["labels"].to(device)
                loss_weights = chunk["loss_weights"].to(device)
                model_inputs = {
                    "input_ids": chunk["input_ids"].to(device),
                    "attention_mask": chunk["attention_mask"].to(device),
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    logits = model(**model_inputs).logits
                    loss = risk_weighted_cross_entropy(logits, labels, loss_weights)
                (loss / chunk_count).backward()
                batch_loss += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(stage.get("gradient_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            updates += 1
            running_loss += batch_loss / chunk_count
        completed_epochs = epoch + 1
        if updates >= total_updates:
            break

    if distributed:
        loss_tensor = torch.tensor([running_loss, updates], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        aggregate_loss = float(loss_tensor[0].item())
        aggregate_updates = int(loss_tensor[1].item())
    else:
        aggregate_loss = running_loss
        aggregate_updates = updates

    output = Path(output_dir)
    summary = {
        "training_rows": len(rows),
        "optimizer_updates_per_process": updates,
        "completed_epochs": completed_epochs,
        "world_size": world_size,
        "mean_training_loss": aggregate_loss / max(1, aggregate_updates),
        "seed": seed,
        "stage": dict(stage),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
    }
    if rank == 0:
        unwrapped = model.module if distributed else model
        _atomic_torch_save({"model_state": unwrapped.state_dict()}, output / "checkpoint.pt")
        write_json_atomic(output / "training_summary.json", summary)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return summary
