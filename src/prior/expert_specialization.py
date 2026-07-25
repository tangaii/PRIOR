"""Deterministic curricula for complementary expert specialization.

The coverage curriculum emphasizes rare, critical, long, and low-density
records.  The precision curriculum emphasizes no-PII and format-confusable
records while retaining high-risk positives.  Both curricula consume the
shared records produced by :mod:`prior.risk_aware_span_tagging`.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .risk_aware_span_tagging import prepare_training_record
from .schema import normalize_label


LONG_TEXT_CHARACTERS = 500
LOW_DENSITY_MAX_SPANS = 2
LOW_DENSITY_MAX_RATIO = 0.02

DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b"
)
NUMERIC_PATTERN = re.compile(r"\b(?:\d[\d\- ]{5,}\d)\b")
GEOGRAPHIC_PATTERN = re.compile(
    r"\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|city|county|"
    r"state|province|zip|postal|district|north|south|east|west)\b",
    re.IGNORECASE,
)
AMBIGUOUS_ENTITY_PATTERN = re.compile(
    r"\b(?:university|bank|hospital|inc\.?|corp\.?|company|agency|department|"
    r"team|school|office|committee)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CurriculumSpec:
    """Capacity and deterministic priority order for one curriculum."""

    name: str
    target_size: int
    quotas: Mapping[str, int]
    priority_order: Sequence[str]


def stable_sample_rank(sample_id: str) -> int:
    """Return the stable rank used by the executed curriculum builders."""

    return int(hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()[:16], 16)


class StableReservoir:
    """Retain the lowest stable hashes without depending on stream order."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._items: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, row: Mapping[str, Any]) -> None:
        if self.limit <= 0:
            return
        copied = dict(row)
        sample_id = str(copied["sample_id"])
        rank = stable_sample_rank(sample_id)
        entry = (-rank, sample_id, copied)
        if len(self._items) < self.limit:
            heapq.heappush(self._items, entry)
        elif -self._items[0][0] > rank:
            heapq.heapreplace(self._items, entry)

    def values(self) -> list[dict[str, Any]]:
        return [row for _, _, row in sorted(self._items, key=lambda item: (-item[0], item[1]))]


class CurriculumBuilder:
    """Build a quota-prioritized curriculum in one streaming pass."""

    def __init__(self, spec: CurriculumSpec):
        self.spec = spec
        self._reservoirs = {
            bucket: StableReservoir(limit) for bucket, limit in spec.quotas.items()
        }

    def add(self, buckets: Iterable[str], row: Mapping[str, Any]) -> None:
        selected_buckets = set(buckets)
        selected_buckets.add("filler")
        for bucket in selected_buckets:
            reservoir = self._reservoirs.get(bucket)
            if reservoir is not None:
                reservoir.add(row)

    def finalize(self) -> list[dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for bucket in self.spec.priority_order:
            for row in self._reservoirs[bucket].values():
                selected[str(row["sample_id"])] = row
                if len(selected) >= self.spec.target_size:
                    return list(selected.values())[: self.spec.target_size]
        return list(selected.values())[: self.spec.target_size]


def is_low_density(text: str, spans: Sequence[Mapping[str, Any]]) -> bool:
    if not spans:
        return False
    span_characters = sum(
        max(0, int(span["end"]) - int(span["start"])) for span in spans
    )
    return (
        len(spans) <= LOW_DENSITY_MAX_SPANS
        or span_characters / max(1, len(text)) <= LOW_DENSITY_MAX_RATIO
    )


def curriculum_buckets(text: str, spans: Sequence[Mapping[str, Any]]) -> set[str]:
    """Assign all positive and confusable-negative buckets used in training."""

    labels = {normalize_label(span.get("label")) for span in spans}
    risk_bands = {normalize_label(span.get("risk_band")) for span in spans}
    buckets = {
        name
        for label, name in {
            "HEALTH_MEDICAL": "health",
            "AUTH_SECRET": "auth",
            "OFFICIAL_ID": "official",
            "FINANCIAL_ACCOUNT": "financial",
            "DIGITAL_ID": "digital",
            "CONTACT": "contact",
            "DEMOGRAPHIC_PROFILE": "demographic",
            "ADDRESS_GEO": "address",
            "PERSON_NAME": "person",
            "TRANSACTION_ASSET": "transaction",
        }.items()
        if label in labels
    }
    if any(bool(span.get("critical_for_pcr")) for span in spans) or "CRITICAL" in risk_bands:
        buckets.add("critical_risk")
    if risk_bands & {"CRITICAL", "HIGH"}:
        buckets.add("high_risk")
    if len(text) >= LONG_TEXT_CHARACTERS:
        buckets.add("long_text")
    if not spans:
        buckets.add("no_pii")
    if is_low_density(text, spans):
        buckets.add("low_density")

    date_negative = bool(DATE_PATTERN.search(text)) and not (
        labels & {"DATE_TIME", "OFFICIAL_ID"}
    )
    numeric_negative = bool(NUMERIC_PATTERN.search(text)) and not (
        labels
        & {
            "OFFICIAL_ID",
            "FINANCIAL_ACCOUNT",
            "AUTH_SECRET",
            "TRANSACTION_ASSET",
            "CONTACT",
        }
    )
    geographic_negative = bool(GEOGRAPHIC_PATTERN.search(text)) and "ADDRESS_GEO" not in labels
    ambiguous_negative = bool(AMBIGUOUS_ENTITY_PATTERN.search(text)) and not (
        labels & {"PERSON_NAME", "ACCOUNT_HANDLE", "ADDRESS_GEO"}
    )
    for condition, bucket in (
        (date_negative, "date_negative"),
        (numeric_negative, "numeric_negative"),
        (geographic_negative, "geographic_negative"),
        (ambiguous_negative, "ambiguous_negative"),
    ):
        if condition:
            buckets.add(bucket)
    if buckets & {
        "date_negative",
        "numeric_negative",
        "geographic_negative",
        "ambiguous_negative",
    }:
        buckets.add("hard_negative")
    if labels & {"PERSON_NAME", "CONTACT", "ADDRESS_GEO", "OFFICIAL_ID", "HEALTH_MEDICAL"}:
        buckets.add("common")
    return buckets


def shared_initialization_spec() -> CurriculumSpec:
    return CurriculumSpec(
        name="shared_initialization",
        target_size=300_000,
        quotas={
            "health": 18_000,
            "transaction": 3_000,
            "auth": 25_000,
            "financial": 40_000,
            "official": 55_000,
            "digital": 35_000,
            "demographic": 80_000,
            "critical_risk": 90_000,
            "long_text": 60_000,
            "no_pii": 30_000,
            "common": 150_000,
            "filler": 300_000,
        },
        priority_order=(
            "health",
            "transaction",
            "auth",
            "financial",
            "official",
            "digital",
            "demographic",
            "critical_risk",
            "long_text",
            "no_pii",
            "common",
            "filler",
        ),
    )


def shared_continuation_spec() -> CurriculumSpec:
    return CurriculumSpec(
        name="shared_continuation",
        target_size=800_000,
        quotas={
            "health": 22_434,
            "transaction": 3_143,
            "auth": 55_000,
            "financial": 110_000,
            "official": 160_000,
            "digital": 90_000,
            "demographic": 220_000,
            "critical_risk": 360_000,
            "long_text": 160_000,
            "no_pii": 90_000,
            "common": 420_000,
            "filler": 800_000,
        },
        priority_order=(
            "health",
            "transaction",
            "auth",
            "financial",
            "official",
            "digital",
            "critical_risk",
            "demographic",
            "long_text",
            "no_pii",
            "common",
            "filler",
        ),
    )


def coverage_expert_spec() -> CurriculumSpec:
    return CurriculumSpec(
        name="coverage_expert",
        target_size=500_000,
        quotas={
            "demographic": 90_000,
            "address": 70_000,
            "person": 70_000,
            "contact": 80_000,
            "official": 90_000,
            "health": 50_000,
            "auth": 55_000,
            "financial": 70_000,
            "digital": 55_000,
            "transaction": 25_000,
            "critical_risk": 120_000,
            "high_risk": 150_000,
            "no_pii": 60_000,
            "low_density": 120_000,
            "long_text": 90_000,
            "common": 220_000,
            "filler": 500_000,
        },
        priority_order=(
            "demographic",
            "address",
            "person",
            "contact",
            "official",
            "health",
            "auth",
            "financial",
            "digital",
            "transaction",
            "critical_risk",
            "high_risk",
            "no_pii",
            "low_density",
            "long_text",
            "common",
            "filler",
        ),
    )


def precision_expert_spec() -> CurriculumSpec:
    return CurriculumSpec(
        name="precision_expert",
        target_size=300_000,
        quotas={
            "hard_negative": 130_000,
            "date_negative": 50_000,
            "numeric_negative": 60_000,
            "geographic_negative": 50_000,
            "ambiguous_negative": 40_000,
            "no_pii": 100_000,
            "low_density": 120_000,
            "long_text": 70_000,
            "critical_risk": 45_000,
            "high_risk": 60_000,
            "official": 25_000,
            "health": 20_000,
            "auth": 20_000,
            "contact": 25_000,
            "common": 80_000,
            "filler": 300_000,
        },
        priority_order=(
            "hard_negative",
            "date_negative",
            "numeric_negative",
            "geographic_negative",
            "ambiguous_negative",
            "no_pii",
            "low_density",
            "long_text",
            "critical_risk",
            "high_risk",
            "official",
            "health",
            "auth",
            "contact",
            "common",
            "filler",
        ),
    )


def anchor_repair_spec() -> CurriculumSpec:
    """Return the public high-risk, low-FP anchor-repair curriculum.

    The anchor keeps critical positives early while retaining format-confusable
    and no-PII rows. This makes the public retraining protocol self-contained
    without depending on unpublished row manifests.
    """

    return CurriculumSpec(
        name="anchor_repair",
        target_size=200_000,
        quotas={
            "critical_risk": 75_000,
            "high_risk": 80_000,
            "health": 25_000,
            "auth": 25_000,
            "financial": 35_000,
            "official": 40_000,
            "digital": 30_000,
            "hard_negative": 70_000,
            "no_pii": 55_000,
            "low_density": 60_000,
            "long_text": 45_000,
            "common": 90_000,
            "filler": 200_000,
        },
        priority_order=(
            "critical_risk",
            "health",
            "auth",
            "financial",
            "official",
            "digital",
            "high_risk",
            "hard_negative",
            "no_pii",
            "low_density",
            "long_text",
            "common",
            "filler",
        ),
    )


def curriculum_specs() -> dict[str, CurriculumSpec]:
    specs = (
        shared_initialization_spec(),
        shared_continuation_spec(),
        coverage_expert_spec(),
        precision_expert_spec(),
        anchor_repair_spec(),
    )
    return {spec.name: spec for spec in specs}


def build_curricula(
    rows: Iterable[Mapping[str, Any]],
    specs: Sequence[CurriculumSpec],
    *,
    excluded_sample_ids: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build multiple curricula in one pass over the public training data."""

    excluded_sample_ids = excluded_sample_ids or set()
    builders = {spec.name: CurriculumBuilder(spec) for spec in specs}
    ignored = Counter()
    scanned = accepted = 0
    for source_row in rows:
        scanned += 1
        sample_id = str(source_row["sample_id"])
        if sample_id in excluded_sample_ids:
            ignored["excluded_sample"] += 1
            continue
        record = prepare_training_record(source_row)
        buckets = curriculum_buckets(record["text"], record["train_spans"])
        for builder in builders.values():
            builder.add(buckets, record)
        accepted += 1
    curricula = {
        name: sorted(builder.finalize(), key=lambda row: stable_sample_rank(str(row["sample_id"])))
        for name, builder in builders.items()
    }
    summary = {
        "scanned_rows": scanned,
        "accepted_rows": accepted,
        "excluded_rows": int(ignored["excluded_sample"]),
        "curriculum_rows": {name: len(records) for name, records in curricula.items()},
    }
    return curricula, summary
