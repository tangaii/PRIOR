#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PII-PolicyBench risk-aware PCR evaluator.

Metric definition:
1. `weighted_pii_coverage` covers `LOW / MEDIUM / HIGH / CRITICAL` spans.
2. Sample-level mandatory coverage only uses `CRITICAL` spans.
3. `fp_char_rate` constrains over-redaction of non-PII characters.
4. `critical_for_pcr` is derived from `risk_band == CRITICAL` when building gold spans.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PII-PolicyBench PCR from prediction JSONL.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing gold JSONL files.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True, help="Submission JSONL with sample_id and pred_spans.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--split-names", default="", help="Comma-separated file stems to evaluate, e.g. train,test.")
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    parser.add_argument("--critical-threshold", type=float, default=1.0)
    parser.add_argument("--fp-char-rate-threshold", type=float, default=0.05)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--worst-top-k", type=int, default=20)
    return parser.parse_args()


def normalize_label(label: Any) -> str:
    """Normalize label strings."""
    return str(label or "").strip().upper().replace(" ", "_").replace("-", "_")


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    if den == 0:
        return default
    return num / den


def risk_band_to_pcr_weight(risk_band: Any) -> float:
    """Map a risk band to the frozen PCR weight."""
    band = normalize_label(risk_band)
    if band == "CRITICAL":
        return 4.0
    if band == "HIGH":
        return 3.0
    if band == "MEDIUM":
        return 2.0
    if band == "LOW":
        return 1.0
    return 0.0


def is_critical_for_pcr(risk_band: Any) -> bool:
    """Return whether a risk band is mandatory for PCR."""
    return normalize_label(risk_band) == "CRITICAL"


def span_bounds(span: Dict[str, Any], text_len: int) -> Tuple[int, int]:
    """Clamp span boundaries to the text length."""
    start = max(0, min(int(span.get("start", span.get("start_char", 0))), text_len))
    end = max(0, min(int(span.get("end", span.get("end_char", 0))), text_len))
    if end < start:
        end = start
    return start, end


def is_positive_span(span: Dict[str, Any]) -> bool:
    """Return whether a span is included in PCR coverage."""
    if span.get("pcr_in_scope") is not None:
        return bool(span["pcr_in_scope"])
    if span.get("weight") is not None:
        return float(span["weight"]) > 0
    return False


def span_weight(span: Dict[str, Any]) -> float:
    """Read the PCR weight from a span."""
    if span.get("weight") is not None:
        return float(span["weight"])
    if span.get("pcr_weight") is not None:
        return float(span["pcr_weight"])
    return 0.0


def span_overlap(pred_span: Dict[str, Any], gt_span: Dict[str, Any], tolerance: int) -> Tuple[bool, int, int]:
    """Return whether two spans match, plus overlap and gap."""
    pred_start, pred_end = int(pred_span["start"]), int(pred_span["end"])
    gt_start, gt_end = int(gt_span["start"]), int(gt_span["end"])

    overlap = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    if overlap > 0:
        return True, overlap, 0
    gap = min(abs(pred_end - gt_start), abs(gt_end - pred_start))
    return gap <= tolerance, 0, gap


def match_spans(
    gt_spans: Sequence[Dict[str, Any]],
    pred_spans: Sequence[Dict[str, Any]],
    tolerance: int,
    label_aware: bool,
) -> Tuple[Set[int], Set[int]]:
    """Greedily match gold spans to predicted spans."""
    valid_gt = [(idx, span) for idx, span in enumerate(gt_spans) if is_positive_span(span)]
    valid_pred = [(idx, span) for idx, span in enumerate(pred_spans) if is_positive_span(span)]
    valid_gt.sort(key=lambda item: -span_weight(item[1]))

    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()
    for gt_idx, gt_span in valid_gt:
        gt_label = normalize_label(gt_span.get("label"))
        best_pred_idx: Optional[int] = None
        best_rank: Optional[Tuple[int, int, float]] = None
        for pred_idx, pred_span in valid_pred:
            if pred_idx in matched_pred:
                continue
            if label_aware and normalize_label(pred_span.get("label")) != gt_label:
                continue
            matched, overlap, gap = span_overlap(pred_span, gt_span, tolerance=tolerance)
            if not matched:
                continue
            rank = (overlap, -gap, span_weight(pred_span))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
    return matched_gt, matched_pred


def build_char_mask(text_len: int, spans: Sequence[Dict[str, Any]]) -> Set[int]:
    """Build a character-level mask for PCR spans."""
    mask: Set[int] = set()
    for span in spans:
        if not is_positive_span(span):
            continue
        start, end = span_bounds(span, text_len)
        mask.update(range(start, end))
    return mask


def char_metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute character-level precision, recall, F1, and IoU."""
    precision = safe_div(tp, tp + fp, default=1.0)
    recall = safe_div(tp, tp + fn, default=1.0)
    f1 = safe_div(2 * precision * recall, precision + recall, default=0.0)
    iou = safe_div(tp, tp + fp + fn, default=1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def compute_sample_metrics(
    sample_id: str,
    dataset: str,
    text: str,
    gt_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    coverage_threshold: float,
    critical_threshold: float,
    fp_char_rate_threshold: float,
    tolerance: int,
) -> Dict[str, Any]:
    """Compute PCR component metrics for one sample."""
    text_len = len(text)
    gt_spans = [span for span in gt_spans if is_positive_span(span)]
    pred_spans = [span for span in pred_spans if is_positive_span(span)]

    matched_gt, matched_pred = match_spans(gt_spans=gt_spans, pred_spans=pred_spans, tolerance=tolerance, label_aware=True)

    gt_weight = sum(span_weight(span) for span in gt_spans)
    pred_weight = sum(span_weight(span) for span in pred_spans)
    matched_gt_weight = sum(span_weight(gt_spans[idx]) for idx in matched_gt)
    matched_pred_weight = sum(span_weight(pred_spans[idx]) for idx in matched_pred)

    critical_gt_indices = [idx for idx, span in enumerate(gt_spans) if bool(span.get("critical_for_pcr"))]
    critical_gt_weight = sum(span_weight(gt_spans[idx]) for idx in critical_gt_indices)
    matched_critical_gt_weight = sum(span_weight(gt_spans[idx]) for idx in critical_gt_indices if idx in matched_gt)

    weighted_coverage = safe_div(matched_gt_weight, gt_weight, default=1.0)
    weighted_precision = safe_div(matched_pred_weight, pred_weight, default=1.0)
    critical_coverage = safe_div(matched_critical_gt_weight, critical_gt_weight, default=1.0)
    risk_adjusted_f2 = safe_div(5.0 * weighted_precision * weighted_coverage, 4.0 * weighted_precision + weighted_coverage, default=0.0)

    gt_mask = build_char_mask(text_len, gt_spans)
    pred_mask = build_char_mask(text_len, pred_spans)
    tp_chars = len(gt_mask & pred_mask)
    fp_chars = len(pred_mask - gt_mask)
    fn_chars = len(gt_mask - pred_mask)
    non_pii_chars = max(0, text_len - len(gt_mask))
    fp_char_rate = safe_div(fp_chars, non_pii_chars, default=0.0)
    non_pii_preservation = 1.0 - fp_char_rate

    coverage_pass = weighted_coverage >= coverage_threshold
    critical_pass = critical_coverage >= critical_threshold
    fp_char_pass = fp_char_rate <= fp_char_rate_threshold
    policy_pass = coverage_pass and critical_pass and fp_char_pass

    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "text": text,
        "gt_spans": gt_spans,
        "pred_spans": pred_spans,
        "text_len": text_len,
        "gt_count": len(gt_spans),
        "pred_count": len(pred_spans),
        "gt_weight": gt_weight,
        "pred_weight": pred_weight,
        "matched_gt_weight": matched_gt_weight,
        "matched_pred_weight": matched_pred_weight,
        "critical_gt_weight": critical_gt_weight,
        "matched_critical_gt_weight": matched_critical_gt_weight,
        "weighted_pii_coverage": weighted_coverage,
        "weighted_precision": weighted_precision,
        "critical_coverage": critical_coverage,
        "risk_adjusted_f2": risk_adjusted_f2,
        "gt_char_count": len(gt_mask),
        "pred_char_count": len(pred_mask),
        "tp_char_count": tp_chars,
        "fp_char_count": fp_chars,
        "fn_char_count": fn_chars,
        "non_pii_char_count": non_pii_chars,
        "fp_char_rate": fp_char_rate,
        "non_pii_preservation": non_pii_preservation,
        "coverage_pass": coverage_pass,
        "critical_pass": critical_pass,
        "fp_char_pass": fp_char_pass,
        "policy_pass": policy_pass,
    }


def aggregate_macro(rows: Sequence[Dict[str, Any]], key: str) -> float:
    """Compute a sample-level macro average."""
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / float(len(rows))


def aggregate_micro(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute corpus-level micro averages."""
    gt_weight = sum(row["gt_weight"] for row in rows)
    pred_weight = sum(row["pred_weight"] for row in rows)
    matched_gt_weight = sum(row["matched_gt_weight"] for row in rows)
    matched_pred_weight = sum(row["matched_pred_weight"] for row in rows)
    precision = safe_div(matched_pred_weight, pred_weight, default=1.0)
    recall = safe_div(matched_gt_weight, gt_weight, default=1.0)
    f1 = safe_div(2 * precision * recall, precision + recall, default=0.0)
    return {
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }


def exact_entity_summary(details: Sequence[Dict[str, Any]], label_aware: bool) -> Dict[str, Any]:
    """Compute exact entity-level precision, recall, and F1."""
    tp = fp = fn = 0
    per_label: Counter[str] = Counter()
    for row in details:
        gt_spans = row["gt_spans"]
        pred_spans = row["pred_spans"]
        matched_gt, matched_pred = match_spans(gt_spans, pred_spans, tolerance=0, label_aware=label_aware)
        tp += len(matched_gt)
        fp += len(pred_spans) - len(matched_pred)
        fn += len(gt_spans) - len(matched_gt)
        for idx in matched_gt:
            per_label[normalize_label(gt_spans[idx]["label"])] += 1
    precision = safe_div(tp, tp + fp, default=1.0)
    recall = safe_div(tp, tp + fn, default=1.0)
    f1 = safe_div(2 * precision * recall, precision + recall, default=0.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_label_tp": dict(sorted(per_label.items())),
    }


def entity_level_summary(details: Sequence[Dict[str, Any]], tolerance: int, label_aware: bool) -> Dict[str, Any]:
    """Compute overlap-based entity-level precision, recall, and F1."""
    tp = fp = fn = 0
    for row in details:
        matched_gt, matched_pred = match_spans(row["gt_spans"], row["pred_spans"], tolerance=tolerance, label_aware=label_aware)
        tp += len(matched_gt)
        fp += len(row["pred_spans"]) - len(matched_pred)
        fn += len(row["gt_spans"]) - len(matched_gt)
    precision = safe_div(tp, tp + fp, default=1.0)
    recall = safe_div(tp, tp + fn, default=1.0)
    f1 = safe_div(2 * precision * recall, precision + recall, default=0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def build_gold_spans(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert source-format or release-format annotations to evaluator gold spans."""
    if "input" not in record:
        out = []
        for ann in record.get("pii_annotations", []):
            risk_band = normalize_label(ann.get("risk_band"))
            pcr_weight = ann.get("weight", ann.get("pcr_weight"))
            if pcr_weight is None:
                pcr_weight = risk_band_to_pcr_weight(risk_band)
            pcr_in_scope = ann.get("pcr_in_scope")
            if pcr_in_scope is None:
                pcr_in_scope = float(pcr_weight) > 0.0
            critical_for_pcr = ann.get("critical_for_pcr")
            if critical_for_pcr is None:
                critical_for_pcr = is_critical_for_pcr(risk_band)
            out.append(
                {
                    "label": normalize_label(ann.get("label")),
                    "fine_labels": list(ann.get("fine_labels", [])),
                    "risk_band": risk_band,
                    "start": int(ann.get("start", ann.get("start_char", 0))),
                    "end": int(ann.get("end", ann.get("end_char", 0))),
                    "weight": float(pcr_weight or 0.0),
                    "critical_for_pcr": bool(critical_for_pcr),
                    "pcr_in_scope": bool(pcr_in_scope),
                }
            )
        return out

    out = []
    for ann in record["input"].get("pii_annotations", []):
        risk_band = normalize_label(ann.get("risk_band"))
        pcr_weight = ann.get("pcr_weight")
        if pcr_weight is None:
            pcr_weight = risk_band_to_pcr_weight(risk_band)
        pcr_in_scope = ann.get("pcr_in_scope")
        if pcr_in_scope is None:
            pcr_in_scope = float(pcr_weight) > 0.0
        critical_for_pcr = is_critical_for_pcr(risk_band)
        out.append(
            {
                "label": normalize_label(ann.get("coarse_label")),
                "fine_label": ann.get("normalized_fine_label"),
                "start": int(ann["start_char"]),
                "end": int(ann["end_char"]),
                "weight": float(pcr_weight or 0.0),
                "critical_for_pcr": critical_for_pcr,
                "pcr_in_scope": bool(pcr_in_scope),
            }
        )
    return out


def normalize_prediction_spans(pred_spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize minimal challenge submission spans.

    Participants only need to submit `start`, `end`, and `label`. If `weight`
    or `pcr_in_scope` is present, it is preserved; otherwise each prediction is
    treated as an in-scope positive span with unit weight.
    """
    normalized: List[Dict[str, Any]] = []
    for span in pred_spans:
        out = dict(span)
        out["label"] = normalize_label(out.get("label"))
        out["start"] = int(out.get("start", out.get("start_char", 0)))
        out["end"] = int(out.get("end", out.get("end_char", out["start"])))
        if out.get("weight") is None:
            out["weight"] = 1.0
        if out.get("pcr_in_scope") is None:
            out["pcr_in_scope"] = True
        normalized.append(out)
    return normalized


def read_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read predictions and index them by sample_id."""
    pred_by_sample: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Support both challenge submissions and historical baseline prediction files.
            row["pred_spans"] = normalize_prediction_spans(row.get("pred_spans", []))
            pred_by_sample[str(row["sample_id"])] = row
    return pred_by_sample


def canonical_sample_id(record: Dict[str, Any], rel_path: Path, line_idx: int) -> str:
    """Build a stable sample_id for standalone source-dataset evaluation."""
    if record.get("sample_id") is not None:
        return str(record["sample_id"])
    meta = record.get("meta") or {}
    dataset_name = str(meta.get("dataset_dir") or (rel_path.parts[0] if rel_path.parts else "unknown"))
    source_split = str(meta.get("source_split") or rel_path.stem)
    source_sample_id = meta.get("source_sample_id")
    if source_sample_id is not None:
        return f"{dataset_name}:{source_split}:{source_sample_id}"
    return f"{rel_path}:{line_idx}"


def parse_split_names(raw: str) -> Set[str]:
    """Parse a comma-separated list of split names."""
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def iter_dataset_records(root: Path, split_names: Optional[Set[str]] = None) -> Iterator[Tuple[Path, str, Dict[str, Any]]]:
    """Iterate over source-format or release-format gold records."""
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    for path in paths:
        if split_names and path.stem not in split_names:
            continue
        rel = path.name if root.is_file() else path.relative_to(root)
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = canonical_sample_id(record=row, rel_path=Path(rel), line_idx=idx)
                yield path, sample_id, row


def record_text(record: Dict[str, Any]) -> str:
    """Return the text field from source-format or release-format records."""
    if "input" in record:
        return str(record["input"].get("raw_text") or "")
    return str(record.get("text") or "")


def record_dataset(record: Dict[str, Any], path: Path) -> str:
    """Return a dataset/source label for grouping metrics."""
    if record.get("dataset") is not None:
        return str(record["dataset"])
    if "meta" in record:
        return str((record.get("meta") or {}).get("dataset_dir") or "unknown")
    sample_id = str(record.get("sample_id") or "")
    if ":" in sample_id:
        return sample_id.split(":", 1)[0]
    return path.stem or "unknown"


def summarize_evaluation_results(
    details: Sequence[Dict[str, Any]],
    by_dataset: Dict[str, List[Dict[str, Any]]],
    coverage_threshold: float,
    critical_threshold: float,
    fp_char_rate_threshold: float,
    tolerance: int,
    worst_top_k: int,
) -> Dict[str, Any]:
    """Aggregate sample-level results into the final summary."""
    positive_rows = [row for row in details if row["gt_count"] > 0]
    negative_rows = [row for row in details if row["gt_count"] == 0]

    sample_macro = {
        "policy_compliance_rate": aggregate_macro(details, "policy_pass"),
        "weighted_pii_coverage": aggregate_macro(details, "weighted_pii_coverage"),
        "weighted_precision": aggregate_macro(details, "weighted_precision"),
        "critical_coverage": aggregate_macro(details, "critical_coverage"),
        "fp_char_rate": aggregate_macro(details, "fp_char_rate"),
        "non_pii_preservation": aggregate_macro(details, "non_pii_preservation"),
        "risk_adjusted_f2": aggregate_macro(details, "risk_adjusted_f2"),
    }
    positive_macro = {
        "n_samples": len(positive_rows),
        "policy_compliance_rate": aggregate_macro(positive_rows, "policy_pass"),
        "weighted_pii_coverage": aggregate_macro(positive_rows, "weighted_pii_coverage"),
        "weighted_precision": aggregate_macro(positive_rows, "weighted_precision"),
        "critical_coverage": aggregate_macro(positive_rows, "critical_coverage"),
        "fp_char_rate": aggregate_macro(positive_rows, "fp_char_rate"),
        "risk_adjusted_f2": aggregate_macro(positive_rows, "risk_adjusted_f2"),
    }
    negative_macro = {
        "n_samples": len(negative_rows),
        "policy_compliance_rate": aggregate_macro(negative_rows, "policy_pass"),
        "fp_char_rate": aggregate_macro(negative_rows, "fp_char_rate"),
        "non_pii_preservation": aggregate_macro(negative_rows, "non_pii_preservation"),
    }

    micro = aggregate_micro(details)
    char_tp = sum(row["tp_char_count"] for row in details)
    char_fp = sum(row["fp_char_count"] for row in details)
    char_fn = sum(row["fn_char_count"] for row in details)
    char_tn = sum(max(0, row["non_pii_char_count"] - row["fp_char_count"]) for row in details)
    total_chars = sum(row["text_len"] for row in details)
    char_level = {
        "tp_chars": char_tp,
        "fp_chars": char_fp,
        "fn_chars": char_fn,
        "tn_chars": char_tn,
        **char_metrics_from_counts(char_tp, char_fp, char_fn),
    }
    char_level["accuracy"] = safe_div(char_tp + char_tn, total_chars, default=1.0)

    by_dataset_summary = {}
    for dataset, rows in sorted(by_dataset.items()):
        by_dataset_summary[dataset] = {
            "n_samples": len(rows),
            "policy_compliance_rate": aggregate_macro(rows, "policy_pass"),
            "weighted_pii_coverage": aggregate_macro(rows, "weighted_pii_coverage"),
            "weighted_precision": aggregate_macro(rows, "weighted_precision"),
            "critical_coverage": aggregate_macro(rows, "critical_coverage"),
            "fp_char_rate": aggregate_macro(rows, "fp_char_rate"),
        }

    worst_cases = sorted(
        details,
        key=lambda row: (
            row["policy_pass"],
            row["critical_coverage"],
            row["weighted_pii_coverage"],
            -row["fp_char_rate"],
        ),
    )[:worst_top_k]

    return {
        "policy": {
            "coverage_threshold": coverage_threshold,
            "critical_threshold": critical_threshold,
            "fp_char_rate_threshold": fp_char_rate_threshold,
            "tolerance": tolerance,
        },
        "n_samples": len(details),
        "sample_macro": sample_macro,
        "positive_only_macro": positive_macro,
        "negative_only_macro": negative_macro,
        "corpus_micro": micro,
        "char_level": char_level,
        "entity_level": {
            "exact_label_aware": exact_entity_summary(details, label_aware=True),
            "overlap_label_agnostic": entity_level_summary(details, tolerance=tolerance, label_aware=False),
            "overlap_label_aware": entity_level_summary(details, tolerance=tolerance, label_aware=True),
        },
        "by_dataset": by_dataset_summary,
        "worst_cases": worst_cases,
        "details": list(details),
    }


def evaluate_predictions(
    data_root: Path,
    pred_by_sample: Dict[str, Dict[str, Any]],
    split_names: Optional[Set[str]],
    coverage_threshold: float,
    critical_threshold: float,
    fp_char_rate_threshold: float,
    tolerance: int,
    worst_top_k: int,
) -> Dict[str, Any]:
    """Run standalone source-dataset evaluation."""
    details: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path, sample_id, record in iter_dataset_records(data_root, split_names=split_names):
        pred_row = pred_by_sample.get(sample_id, {"sample_id": sample_id, "pred_spans": []})
        text = record_text(record)
        dataset = record_dataset(record, path)
        gold = build_gold_spans(record)
        pred_spans = pred_row.get("pred_spans", [])
        detail = compute_sample_metrics(
            sample_id=sample_id,
            dataset=dataset,
            text=text,
            gt_spans=gold,
            pred_spans=pred_spans,
            coverage_threshold=coverage_threshold,
            critical_threshold=critical_threshold,
            fp_char_rate_threshold=fp_char_rate_threshold,
            tolerance=tolerance,
        )
        details.append(detail)
        by_dataset[dataset].append(detail)

    return summarize_evaluation_results(
        details=details,
        by_dataset=by_dataset,
        coverage_threshold=coverage_threshold,
        critical_threshold=critical_threshold,
        fp_char_rate_threshold=fp_char_rate_threshold,
        tolerance=tolerance,
        worst_top_k=worst_top_k,
    )


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    preds = read_predictions(args.predictions_jsonl)
    split_names = parse_split_names(args.split_names)
    result = evaluate_predictions(
        data_root=args.data_root,
        pred_by_sample=preds,
        split_names=split_names,
        coverage_threshold=args.coverage_threshold,
        critical_threshold=args.critical_threshold,
        fp_char_rate_threshold=args.fp_char_rate_threshold,
        tolerance=args.tolerance,
        worst_top_k=args.worst_top_k,
    )

    out_dir = args.output_dir
    write_json(out_dir / "policy_summary.json", {k: v for k, v in result.items() if k != "details"})
    write_jsonl(out_dir / "policy_details.jsonl", result["details"])
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "n_samples": result["n_samples"],
                "sample_macro": result["sample_macro"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
