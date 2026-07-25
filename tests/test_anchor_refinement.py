from __future__ import annotations

import unittest

from prior.anchor_refinement import refine_anchor_spans


class AnchorRefinementTest(unittest.TestCase):
    def test_threshold_trim_restore_and_exact_deduplication(self) -> None:
        text = "  Alice  !!!"
        spans = [
            {"start": 0, "end": 9, "label": "PERSON_NAME", "score": 0.8},
            {"start": 2, "end": 7, "label": "PERSON_NAME", "score": 0.9},
            {"start": 9, "end": 12, "label": "CONTACT", "score": 0.7},
            {"start": 0, "end": 2, "label": "CONTACT", "score": 0.1},
        ]
        result = refine_anchor_spans(
            text,
            spans,
            thresholds={"PERSON_NAME": 0.5, "CONTACT": 0.5},
            trim_characters=" !",
        )
        self.assertEqual(
            result,
            [
                {"start": 2, "end": 7, "label": "PERSON_NAME"},
                {"start": 9, "end": 12, "label": "CONTACT"},
            ],
        )

    def test_distinct_overlaps_are_retained(self) -> None:
        result = refine_anchor_spans(
            "abcdef",
            [
                {"start": 0, "end": 4, "label": "CONTACT", "score": 1.0},
                {"start": 2, "end": 6, "label": "DIGITAL_ID", "score": 1.0},
            ],
            thresholds={"CONTACT": 0.5, "DIGITAL_ID": 0.5},
            trim_characters=" ",
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
