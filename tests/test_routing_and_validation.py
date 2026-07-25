from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prior.validation import validate_submission
from prior.whole_output_routing import (
    choose_source_routes,
    route_prediction_files,
    select_complete_output,
)


class RoutingAndValidationTest(unittest.TestCase):
    def test_route_gate_and_lexicographic_selection(self) -> None:
        def scope(rows: int, passes: int, pcr: float):
            return {"rows": rows, "passes": passes, "pcr": pcr}

        audit = {
            "alpha": {
                "anchor": {
                    "development": scope(10, 8, 0.8),
                    "fresh": scope(5, 4, 0.8),
                    "all": scope(15, 12, 0.8),
                },
                "coverage": {
                    "development": scope(10, 9, 0.9),
                    "fresh": scope(5, 4, 0.8),
                    "all": scope(15, 13, 0.866),
                },
                "precision": {
                    "development": scope(10, 9, 0.9),
                    "fresh": scope(5, 3, 0.6),
                    "all": scope(15, 12, 0.8),
                },
            }
        }
        self.assertEqual(choose_source_routes(audit), {"alpha": "coverage"})

    def test_selection_copies_one_complete_output(self) -> None:
        rows = {
            "anchor": {"sample_id": "alpha:1", "pred_spans": []},
            "coverage": {
                "sample_id": "alpha:1",
                "pred_spans": [{"start": 0, "end": 2, "label": "CONTACT"}],
            },
            "precision": {
                "sample_id": "alpha:1",
                "pred_spans": [{"start": 3, "end": 5, "label": "PERSON_NAME"}],
            },
        }
        selected = select_complete_output(rows, routes={"alpha": "coverage"})
        self.assertEqual(selected["pred_spans"], rows["coverage"]["pred_spans"])
        self.assertNotIn(rows["precision"]["pred_spans"][0], selected["pred_spans"])

    def test_streaming_router_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_rows = {
                "anchor": [
                    {"sample_id": "alpha:1", "pred_spans": []},
                    {"sample_id": "beta:2", "pred_spans": []},
                ],
                "coverage": [
                    {
                        "sample_id": "alpha:1",
                        "pred_spans": [{"start": 0, "end": 2, "label": "CONTACT"}],
                    },
                    {"sample_id": "beta:2", "pred_spans": []},
                ],
                "precision": [
                    {"sample_id": "alpha:1", "pred_spans": []},
                    {"sample_id": "beta:2", "pred_spans": []},
                ],
            }
            paths = {}
            for expert, rows in source_rows.items():
                path = root / f"{expert}.jsonl"
                path.write_text(
                    "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[expert] = path
            test_path = root / "test.jsonl"
            test_path.write_text(
                json.dumps({"sample_id": "alpha:1", "text": "ab"})
                + "\n"
                + json.dumps({"sample_id": "beta:2", "text": ""})
                + "\n",
                encoding="utf-8",
            )
            output = root / "submission.jsonl"
            stats = route_prediction_files(paths, output, routes={"alpha": "coverage"})
            self.assertEqual(stats["row_count"], 2)
            audit = validate_submission(output, test_path=test_path, expected_rows=2)
            self.assertTrue(audit["strict_pass"])


if __name__ == "__main__":
    unittest.main()
