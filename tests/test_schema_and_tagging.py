from __future__ import annotations

import unittest

from prior.risk_aware_span_tagging import (
    choose_span_for_token,
    decode_bio_spans,
    extract_policy_spans,
)
from prior.schema import OFFICIAL_LABELS, build_bio_label_maps


class SchemaAndTaggingTest(unittest.TestCase):
    def test_bio_schema_has_twenty_three_tags(self) -> None:
        label_to_id, id_to_label = build_bio_label_maps()
        self.assertEqual(len(OFFICIAL_LABELS), 11)
        self.assertEqual(len(label_to_id), 23)
        self.assertEqual(id_to_label[label_to_id["B-AUTH_SECRET"]], "B-AUTH_SECRET")

    def test_policy_span_filtering(self) -> None:
        row = {
            "sample_id": "source:1",
            "pii_annotations": [
                {
                    "start": 0,
                    "end": 4,
                    "label": "PERSON_NAME",
                    "risk_band": "HIGH",
                    "weight": 3,
                    "pcr_in_scope": True,
                },
                {
                    "start": 5,
                    "end": 8,
                    "label": "CONTEXT_MISC",
                    "risk_band": "LOW",
                    "weight": 1,
                    "pcr_in_scope": True,
                },
                {
                    "start": 9,
                    "end": 12,
                    "label": "CONTACT",
                    "risk_band": "LOW",
                    "weight": 0,
                    "pcr_in_scope": True,
                },
            ],
        }
        spans, ignored = extract_policy_spans(row)
        self.assertEqual([span["label"] for span in spans], ["PERSON_NAME"])
        self.assertEqual(ignored["context_misc"], 1)
        self.assertEqual(ignored["zero_weight"], 1)

    def test_overlap_resolution_prioritizes_risk_after_overlap(self) -> None:
        spans = [
            {"start": 0, "end": 8, "label": "PERSON_NAME", "risk_band": "LOW"},
            {"start": 2, "end": 6, "label": "AUTH_SECRET", "risk_band": "CRITICAL"},
        ]
        chosen = choose_span_for_token(2, 5, spans)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["label"], "AUTH_SECRET")

    def test_initial_inside_tag_opens_a_span(self) -> None:
        label_to_id, id_to_label = build_bio_label_maps()
        entries = [
            (2, 5, label_to_id["I-CONTACT"], 0.8),
            (5, 8, label_to_id["I-CONTACT"], 0.6),
            (9, 10, label_to_id["O"], 0.9),
        ]
        self.assertEqual(
            decode_bio_spans(entries, id_to_label),
            [{"start": 2, "end": 8, "label": "CONTACT", "score": 0.7}],
        )


if __name__ == "__main__":
    unittest.main()
