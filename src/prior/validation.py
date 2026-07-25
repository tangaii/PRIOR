"""Strict validation for row-aligned PII-PolicyBench submissions."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .io import iter_jsonl
from .schema import canonical_prediction_span, validate_prediction_span


def validate_submission(
    submission_path: str | Path,
    *,
    test_path: str | Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Validate schema, uniqueness, order, bounds, and optional test alignment."""

    counters = {
        "row_count": 0,
        "duplicate_submission_rows": 0,
        "malformed_span_count": 0,
        "illegal_or_invalid_span_count": 0,
        "duplicate_span_count": 0,
        "sample_id_mismatch_count": 0,
        "missing_prediction_rows": 0,
        "extra_prediction_rows": 0,
    }
    seen_sample_ids: set[str] = set()
    submission_rows = iter_jsonl(submission_path)
    test_rows = iter_jsonl(test_path) if test_path is not None else None
    pairs = zip_longest(submission_rows, test_rows) if test_rows is not None else (
        (row, None) for row in submission_rows
    )
    for submission_row, test_row in pairs:
        if submission_row is None:
            counters["missing_prediction_rows"] += 1
            continue
        if test_path is not None and test_row is None:
            counters["extra_prediction_rows"] += 1
        counters["row_count"] += 1
        sample_id = str(submission_row.get("sample_id"))
        if not sample_id or sample_id == "None":
            counters["sample_id_mismatch_count"] += 1
        if sample_id in seen_sample_ids:
            counters["duplicate_submission_rows"] += 1
        seen_sample_ids.add(sample_id)
        text_length = None
        if test_row is not None:
            if sample_id != str(test_row.get("sample_id")):
                counters["sample_id_mismatch_count"] += 1
            text_length = len(str(test_row.get("text") or ""))
        spans = submission_row.get("pred_spans")
        if not isinstance(spans, list):
            counters["malformed_span_count"] += 1
            continue
        seen_spans: set[tuple[int, int, str]] = set()
        for span in spans:
            if not isinstance(span, Mapping):
                counters["malformed_span_count"] += 1
                continue
            try:
                canonical = canonical_prediction_span(span)
                validate_prediction_span(canonical, text_length=text_length)
            except (KeyError, TypeError, ValueError):
                counters["illegal_or_invalid_span_count"] += 1
                continue
            identity = (canonical["start"], canonical["end"], canonical["label"])
            if identity in seen_spans:
                counters["duplicate_span_count"] += 1
            seen_spans.add(identity)

    if expected_rows is not None and counters["row_count"] != int(expected_rows):
        counters["expected_row_count_mismatch"] = abs(
            counters["row_count"] - int(expected_rows)
        )
    else:
        counters["expected_row_count_mismatch"] = 0
    strict_fields = [key for key in counters if key != "row_count"]
    return {
        **counters,
        "expected_row_count": expected_rows,
        "test_alignment_checked": test_path is not None,
        "strict_pass": all(int(counters[key]) == 0 for key in strict_fields),
    }
