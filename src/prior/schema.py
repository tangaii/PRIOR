"""Shared PII label and prediction schemas."""

from __future__ import annotations

from typing import Any, Mapping


OFFICIAL_LABELS = (
    "PERSON_NAME",
    "ACCOUNT_HANDLE",
    "CONTACT",
    "ADDRESS_GEO",
    "OFFICIAL_ID",
    "FINANCIAL_ACCOUNT",
    "AUTH_SECRET",
    "DIGITAL_ID",
    "DEMOGRAPHIC_PROFILE",
    "HEALTH_MEDICAL",
    "TRANSACTION_ASSET",
)
OFFICIAL_LABEL_SET = frozenset(OFFICIAL_LABELS)

CRITICAL_LABELS = frozenset(
    {
        "OFFICIAL_ID",
        "FINANCIAL_ACCOUNT",
        "AUTH_SECRET",
        "DIGITAL_ID",
        "HEALTH_MEDICAL",
        "TRANSACTION_ASSET",
    }
)

RISK_PRIORITY = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
RISK_MULTIPLIERS = {"CRITICAL": 2.4, "HIGH": 1.8, "MEDIUM": 1.35, "LOW": 1.0}
CATEGORY_WEIGHTS = {
    "O": 0.15,
    "PERSON_NAME": 1.4,
    "ACCOUNT_HANDLE": 1.8,
    "CONTACT": 1.5,
    "ADDRESS_GEO": 1.4,
    "OFFICIAL_ID": 4.0,
    "FINANCIAL_ACCOUNT": 4.5,
    "AUTH_SECRET": 5.0,
    "DIGITAL_ID": 3.0,
    "DEMOGRAPHIC_PROFILE": 2.7,
    "HEALTH_MEDICAL": 6.5,
    "TRANSACTION_ASSET": 3.0,
}
CATEGORY_PRIORITY = {
    "AUTH_SECRET": 100,
    "FINANCIAL_ACCOUNT": 95,
    "OFFICIAL_ID": 92,
    "HEALTH_MEDICAL": 90,
    "CONTACT": 80,
    "DIGITAL_ID": 78,
    "TRANSACTION_ASSET": 72,
    "DEMOGRAPHIC_PROFILE": 65,
    "ACCOUNT_HANDLE": 60,
    "PERSON_NAME": 55,
    "ADDRESS_GEO": 50,
}


def normalize_label(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def build_bio_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    labels = ["O"]
    for label in OFFICIAL_LABELS:
        labels.extend((f"B-{label}", f"I-{label}"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def source_prefix(sample_id: str) -> str:
    return str(sample_id).split(":", 1)[0]


def canonical_prediction_span(span: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start": int(span["start"]),
        "end": int(span["end"]),
        "label": normalize_label(span["label"]),
    }


def validate_prediction_span(span: Mapping[str, Any], text_length: int | None = None) -> None:
    canonical = canonical_prediction_span(span)
    start, end, label = canonical["start"], canonical["end"], canonical["label"]
    if label not in OFFICIAL_LABEL_SET:
        raise ValueError(f"unsupported PII label: {label}")
    if start < 0 or start >= end:
        raise ValueError(f"invalid half-open span: [{start}, {end})")
    if text_length is not None and end > text_length:
        raise ValueError(f"span [{start}, {end}) exceeds text length {text_length}")

