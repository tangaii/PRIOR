from __future__ import annotations

import unittest

from prior.data_preparation import split_assignment
from prior.expert_specialization import (
    CurriculumBuilder,
    CurriculumSpec,
    StableReservoir,
    anchor_repair_spec,
    curriculum_buckets,
    curriculum_specs,
    stable_sample_rank,
)


class SpecializationTest(unittest.TestCase):
    def test_reservoir_is_stream_order_independent(self) -> None:
        rows = [{"sample_id": f"source:{index}"} for index in range(20)]
        forward = StableReservoir(5)
        reverse = StableReservoir(5)
        for row in rows:
            forward.add(row)
        for row in reversed(rows):
            reverse.add(row)
        expected = sorted(rows, key=lambda row: stable_sample_rank(row["sample_id"]))[:5]
        self.assertEqual(forward.values(), reverse.values())
        self.assertEqual(
            {row["sample_id"] for row in forward.values()},
            {row["sample_id"] for row in expected},
        )

    def test_confusable_negative_buckets(self) -> None:
        buckets = curriculum_buckets("Meet at West Street on 2026-07-24", [])
        self.assertIn("no_pii", buckets)
        self.assertIn("date_negative", buckets)
        self.assertIn("geographic_negative", buckets)
        self.assertIn("hard_negative", buckets)

    def test_priority_order_fills_unique_rows(self) -> None:
        spec = CurriculumSpec(
            name="tiny",
            target_size=2,
            quotas={"rare": 2, "filler": 3},
            priority_order=("rare", "filler"),
        )
        builder = CurriculumBuilder(spec)
        builder.add(["rare"], {"sample_id": "source:1"})
        builder.add(["rare"], {"sample_id": "source:2"})
        builder.add([], {"sample_id": "source:3"})
        self.assertEqual(len(builder.finalize()), 2)

    def test_public_split_assignment_is_stable(self) -> None:
        arguments = {
            "seed": 1337,
            "development_fraction": 0.2,
            "heldout_fraction": 0.2,
        }
        assignments = {
            sample_id: split_assignment(sample_id, **arguments)
            for sample_id in ("alpha:1", "alpha:2", "beta:1")
        }
        self.assertEqual(
            assignments,
            {
                sample_id: split_assignment(sample_id, **arguments)
                for sample_id in reversed(("alpha:1", "alpha:2", "beta:1"))
            },
        )

    def test_anchor_repair_is_part_of_public_curricula(self) -> None:
        spec = anchor_repair_spec()
        self.assertEqual(spec.target_size, 200_000)
        self.assertIn("critical_risk", spec.priority_order)
        self.assertIn("hard_negative", spec.priority_order)
        self.assertIn("anchor_repair", curriculum_specs())


if __name__ == "__main__":
    unittest.main()
