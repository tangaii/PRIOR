"""Risk-aware BIO tagging with overlap-aware character-span decoding.

This module implements the first Method component. It keeps the executed
alignment semantics: a token receives ``B`` only when its start offset equals
the annotation start, repeated window probabilities are averaged by identical
character offsets, and an initial ``I`` tag may open a decoded span.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import (
    CATEGORY_PRIORITY,
    CATEGORY_WEIGHTS,
    OFFICIAL_LABEL_SET,
    RISK_MULTIPLIERS,
    RISK_PRIORITY,
    build_bio_label_maps,
    normalize_label,
)


IGNORE_INDEX = -100


def extract_policy_spans(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert public annotations to the 11-category training schema."""

    spans: list[dict[str, Any]] = []
    ignored = {"context_misc": 0, "zero_weight": 0, "out_of_scope": 0, "unknown": 0}
    for annotation in row.get("pii_annotations", []):
        label = normalize_label(annotation.get("label"))
        weight = float(annotation.get("weight", 0.0) or 0.0)
        if label == "CONTEXT_MISC":
            ignored["context_misc"] += 1
            continue
        if not bool(annotation.get("pcr_in_scope")):
            ignored["out_of_scope"] += 1
            continue
        if weight <= 0.0:
            ignored["zero_weight"] += 1
            continue
        if label not in OFFICIAL_LABEL_SET:
            ignored["unknown"] += 1
            continue
        spans.append(
            {
                "start": int(annotation["start"]),
                "end": int(annotation["end"]),
                "label": label,
                "risk_band": normalize_label(annotation.get("risk_band")),
                "weight": weight,
                "critical_for_pcr": bool(annotation.get("critical_for_pcr")),
                "fine_labels": list(annotation.get("fine_labels", [])),
            }
        )
    spans.sort(key=lambda span: (span["start"], span["end"], span["label"]))
    return spans, ignored


def prepare_training_record(row: Mapping[str, Any]) -> dict[str, Any]:
    spans, _ = extract_policy_spans(row)
    return {
        "sample_id": str(row["sample_id"]),
        "text": str(row.get("text") or ""),
        "train_spans": spans,
    }


def choose_span_for_token(
    token_start: int,
    token_end: int,
    spans: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Resolve overlapping annotations using the retained stable priority."""

    best_span: Mapping[str, Any] | None = None
    best_rank: tuple[int, int, int, int] | None = None
    for span in spans:
        overlap = max(
            0,
            min(token_end, int(span["end"])) - max(token_start, int(span["start"])),
        )
        if overlap <= 0:
            continue
        rank = (
            overlap,
            RISK_PRIORITY.get(normalize_label(span.get("risk_band")), 0),
            CATEGORY_PRIORITY.get(normalize_label(span.get("label")), 0),
            int(span["end"]) - int(span["start"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_span = span
    return best_span


def align_training_batch(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    *,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    """Tokenize overflow windows and create weighted BIO targets."""

    import torch

    encoded = tokenizer(
        [str(record.get("text") or "") for record in records],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    overflow = encoded.pop("overflow_to_sample_mapping")
    labels = torch.full(offsets.shape[:2], IGNORE_INDEX, dtype=torch.long)
    loss_weights = torch.zeros(offsets.shape[:2], dtype=torch.float32)

    for window_index, sample_index in enumerate(overflow.tolist()):
        spans = records[sample_index].get("train_spans")
        if spans is None:
            spans, _ = extract_policy_spans(records[sample_index])
        for token_index, (start, end) in enumerate(offsets[window_index].tolist()):
            start, end = int(start), int(end)
            if start == end:
                continue
            chosen = choose_span_for_token(start, end, spans)
            if chosen is None:
                labels[window_index, token_index] = label_to_id["O"]
                loss_weights[window_index, token_index] = CATEGORY_WEIGHTS["O"]
                continue
            label = normalize_label(chosen["label"])
            risk_band = normalize_label(chosen.get("risk_band"))
            prefix = "B" if start == int(chosen["start"]) else "I"
            labels[window_index, token_index] = label_to_id[f"{prefix}-{label}"]
            loss_weights[window_index, token_index] = (
                CATEGORY_WEIGHTS[label] * RISK_MULTIPLIERS[risk_band]
            )

    encoded["labels"] = labels
    encoded["loss_weights"] = loss_weights
    encoded["offset_mapping"] = offsets
    encoded["overflow_to_sample_mapping"] = overflow
    return dict(encoded)


def risk_weighted_cross_entropy(logits: Any, labels: Any, loss_weights: Any) -> Any:
    """Normalized category- and risk-weighted token cross-entropy."""

    import torch.nn.functional as functional

    flat_loss = functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(labels)
    valid = labels.ne(IGNORE_INDEX).float()
    weights = loss_weights * valid
    return (flat_loss * weights).sum() / weights.sum().clamp_min(1.0)


def split_window_batch(batch: Mapping[str, Any], chunk_size: int) -> Iterator[dict[str, Any]]:
    total = int(batch["input_ids"].shape[0])
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield {key: value[start:end] for key, value in batch.items()}


def decode_bio_spans(
    token_entries: Sequence[tuple[int, int, int, float]],
    id_to_label: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Decode averaged token decisions into scored character spans."""

    spans: list[dict[str, Any]] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    scores: list[float] = []

    def flush() -> None:
        nonlocal current_label, current_start, current_end, scores
        if (
            current_label is not None
            and current_start is not None
            and current_end is not None
            and current_start < current_end
        ):
            spans.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "label": current_label,
                    "score": sum(scores) / max(1, len(scores)),
                }
            )
        current_label = None
        current_start = None
        current_end = None
        scores = []

    previous_end: int | None = None
    for start, end, label_id, score in sorted(token_entries, key=lambda item: (item[0], item[1])):
        label = id_to_label[int(label_id)]
        if label == "O":
            flush()
            previous_end = end
            continue
        prefix, entity = label.split("-", 1)
        contiguous = previous_end is not None and start <= previous_end
        if prefix == "B" or current_label != entity or not contiguous:
            flush()
            current_label = entity
            current_start = start
            current_end = end
            scores = [float(score)]
        else:
            current_end = max(int(current_end), int(end))
            scores.append(float(score))
        previous_end = end
    flush()

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        key = (int(span["start"]), int(span["end"]), str(span["label"]))
        if key not in seen:
            seen.add(key)
            deduplicated.append(span)
    return deduplicated


def load_tokenizer(model_dir: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)


def load_tagger(
    model_dir: str | Path,
    *,
    device: Any,
    checkpoint: str | Path | None = None,
    train_mode: bool = False,
) -> Any:
    import torch
    from transformers import AutoModelForTokenClassification

    label_to_id, id_to_label = build_bio_label_maps()
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        ignore_mismatched_sizes=True,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.train(mode=train_mode)
    return model


def predict_scored_spans(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_length: int = 512,
    stride: int = 128,
    batch_size: int = 32,
    span_score_floors: Mapping[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run overlap-window inference and return decoded scored spans."""

    import torch

    _, id_to_label = build_bio_label_maps()
    span_score_floors = span_score_floors or {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            encoded = tokenizer(
                [str(row.get("text") or "") for row in batch_rows],
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            overflow = encoded.pop("overflow_to_sample_mapping")
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**model_inputs).logits, dim=-1).detach().cpu()

            probability_sums: dict[str, dict[tuple[int, int], Any]] = defaultdict(dict)
            counts: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
            for window_index, sample_index in enumerate(overflow.tolist()):
                sample_id = str(batch_rows[sample_index]["sample_id"])
                for token_index, (start, end) in enumerate(offsets[window_index].tolist()):
                    start, end = int(start), int(end)
                    if start == end:
                        continue
                    offset = (start, end)
                    if offset not in probability_sums[sample_id]:
                        probability_sums[sample_id][offset] = probabilities[window_index, token_index].clone()
                        counts[sample_id][offset] = 1
                    else:
                        probability_sums[sample_id][offset] += probabilities[window_index, token_index]
                        counts[sample_id][offset] += 1

            for row in batch_rows:
                sample_id = str(row["sample_id"])
                entries: list[tuple[int, int, int, float]] = []
                for offset, probability_sum in sorted(probability_sums[sample_id].items()):
                    averaged = probability_sum / counts[sample_id][offset]
                    label_id = int(torch.argmax(averaged).item())
                    entries.append((offset[0], offset[1], label_id, float(averaged[label_id].item())))
                decoded = decode_bio_spans(entries, id_to_label)
                predictions[sample_id] = [
                    span
                    for span in decoded
                    if float(span["score"]) >= float(span_score_floors.get(span["label"], 0.0))
                ]
    model.train(mode=was_training)
    return predictions

