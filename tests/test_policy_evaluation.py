from __future__ import annotations

import unittest

from prior.policy_evaluation import compute_sample_metrics


class PolicyEvaluationTest(unittest.TestCase):
    def test_policy_pass_for_covered_critical_span(self) -> None:
        result = compute_sample_metrics(
            sample_id="source:1",
            dataset="source",
            text="secret value",
            gt_spans=[
                {
                    "start": 0,
                    "end": 6,
                    "label": "AUTH_SECRET",
                    "weight": 4.0,
                    "pcr_in_scope": True,
                    "critical_for_pcr": True,
                }
            ],
            pred_spans=[
                {
                    "start": 0,
                    "end": 6,
                    "label": "AUTH_SECRET",
                    "weight": 1.0,
                    "pcr_in_scope": True,
                }
            ],
            coverage_threshold=0.9,
            critical_threshold=1.0,
            fp_char_rate_threshold=0.05,
            tolerance=2,
        )
        self.assertTrue(result["policy_pass"])
        self.assertEqual(result["weighted_pii_coverage"], 1.0)
        self.assertEqual(result["critical_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
