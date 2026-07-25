"""Policy-aligned calibration and safe boundary refinement for the anchor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .schema import normalize_label


def refine_anchor_spans(
    text: str,
    scored_spans: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
    trim_characters: str,
) -> list[dict[str, Any]]:
    """Apply the frozen threshold/trim/fallback/deduplication order.

    Distinct overlaps are retained.  Trimming never expands, splits, merges, or
    relabels a span, and a span that would become empty is restored unchanged.
    """

    trim_set = set(trim_characters)
    retained: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for raw_span in scored_spans:
        label = normalize_label(raw_span.get("label"))
        if label not in thresholds:
            raise ValueError(f"missing anchor threshold for label: {label}")
        if float(raw_span.get("score", 0.0)) < float(thresholds[label]):
            continue
        start, end = int(raw_span["start"]), int(raw_span["end"])
        if start < 0 or start >= end or end > len(text):
            raise ValueError(f"invalid scored span [{start}, {end}) for text length {len(text)}")
        original_start, original_end = start, end
        while start < end and text[start] in trim_set:
            start += 1
        while start < end and text[end - 1] in trim_set:
            end -= 1
        if start >= end:
            start, end = original_start, original_end
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        retained.append({"start": start, "end": end, "label": label})
    return sorted(retained, key=lambda span: (span["start"], span["end"], span["label"]))


def refine_anchor_row(
    row: Mapping[str, Any],
    scored_spans: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
    trim_characters: str,
) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "pred_spans": refine_anchor_spans(
            str(row.get("text") or ""),
            scored_spans,
            thresholds=thresholds,
            trim_characters=trim_characters,
        ),
    }


def threshold_policy_grid(
    baseline: Mapping[str, float],
    *,
    normal_deltas: Iterable[float],
    critical_deltas: Iterable[float],
    critical_labels: set[str],
    floors: Mapping[str, float],
    ceiling: float = 0.95,
) -> list[dict[str, float]]:
    """Generate single-label policies used before held-out policy selection."""

    policies = [dict(baseline)]
    seen = {tuple(sorted(policies[0].items()))}
    for label, initial in baseline.items():
        deltas = critical_deltas if label in critical_labels else normal_deltas
        for delta in deltas:
            candidate = dict(baseline)
            candidate[label] = round(
                max(float(floors[label]), min(float(ceiling), float(initial) + float(delta))),
                4,
            )
            identity = tuple(sorted(candidate.items()))
            if identity not in seen:
                seen.add(identity)
                policies.append(candidate)
    return policies
