"""Source-aware selection of one complete expert prediction set."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .schema import source_prefix


def choose_source_routes(
    audit: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    *,
    anchor_name: str = "anchor",
    expert_order: Sequence[str] = ("coverage", "precision"),
) -> dict[str, str]:
    """Freeze source routes using development gain and fresh non-inferiority."""

    routes: dict[str, str] = {}
    for source, source_audit in sorted(audit.items()):
        anchor = source_audit[anchor_name]
        eligible: list[str] = []
        for expert in expert_order:
            candidate = source_audit[expert]
            development = candidate["development"]
            anchor_development = anchor["development"]
            fresh = candidate["fresh"]
            anchor_fresh = anchor["fresh"]
            improves_development = (
                int(development.get("rows", 0)) > 0
                and float(development["pcr"]) > float(anchor_development["pcr"])
            )
            fresh_noninferior = (
                int(fresh.get("rows", 0)) == 0
                or float(fresh["pcr"]) >= float(anchor_fresh["pcr"])
            )
            if improves_development and fresh_noninferior:
                eligible.append(expert)
        if eligible:
            # Python max preserves expert_order for an exact key tie.
            routes[source] = max(
                eligible,
                key=lambda expert: (
                    float(source_audit[expert]["all"]["pcr"])
                    - float(anchor["all"]["pcr"]),
                    int(source_audit[expert]["all"]["passes"]),
                ),
            )
    return routes


def select_complete_output(
    expert_rows: Mapping[str, Mapping[str, Any]],
    *,
    routes: Mapping[str, str],
    default_expert: str = "anchor",
) -> dict[str, Any]:
    """Copy one expert's complete span list without fusion or modification."""

    sample_ids = {str(row.get("sample_id")) for row in expert_rows.values()}
    if len(sample_ids) != 1 or "None" in sample_ids:
        raise ValueError(f"expert sample_id mismatch: {sample_ids}")
    sample_id = next(iter(sample_ids))
    expert = routes.get(source_prefix(sample_id), default_expert)
    if expert not in expert_rows:
        raise KeyError(f"route selected unavailable expert: {expert}")
    spans = expert_rows[expert].get("pred_spans") or []
    if not isinstance(spans, list):
        raise ValueError(f"pred_spans must be a list for {sample_id}")
    return {"sample_id": sample_id, "pred_spans": spans}


def route_prediction_files(
    expert_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    routes: Mapping[str, str],
    default_expert: str = "anchor",
    compact: bool = True,
) -> dict[str, Any]:
    """Stream row-aligned experts and atomically construct the submission."""

    if default_expert not in expert_paths:
        raise KeyError(f"default expert is missing: {default_expert}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    counts = {expert: 0 for expert in expert_paths}
    changed_from_anchor = 0
    row_count = 0
    separators = (",", ":") if compact else None
    try:
        with ExitStack() as stack:
            handles = {
                expert: stack.enter_context(Path(path).open("r", encoding="utf-8"))
                for expert, path in expert_paths.items()
            }
            output_handle = stack.enter_context(temporary.open("w", encoding="utf-8"))
            for row_count in range(1, 10**12):
                lines = {expert: handle.readline() for expert, handle in handles.items()}
                ended = {expert for expert, line in lines.items() if line == ""}
                if ended:
                    if len(ended) != len(handles):
                        raise RuntimeError(
                            f"expert row-count mismatch after {row_count - 1} rows: ended={sorted(ended)}"
                        )
                    row_count -= 1
                    break
                rows: dict[str, dict[str, Any]] = {}
                for expert, line in lines.items():
                    if not line.strip():
                        raise ValueError(f"blank line in {expert} at row {row_count}")
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"non-object row in {expert} at row {row_count}")
                    rows[expert] = row
                selected = select_complete_output(
                    rows,
                    routes=routes,
                    default_expert=default_expert,
                )
                selected_expert = routes.get(source_prefix(selected["sample_id"]), default_expert)
                counts[selected_expert] += 1
                changed_from_anchor += int(
                    selected_expert != default_expert
                    and selected["pred_spans"] != (rows[default_expert].get("pred_spans") or [])
                )
                output_handle.write(
                    json.dumps(selected, ensure_ascii=False, separators=separators) + "\n"
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "row_count": row_count,
        "source_rows": counts,
        "changed_from_anchor": changed_from_anchor,
        "routes": dict(routes),
    }
