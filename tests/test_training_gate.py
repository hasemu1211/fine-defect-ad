import unittest

from fine_defect_ad.pilot import PILOT_STEPS, READY, expected_pilot_protocol_metadata
from fine_defect_ad.training_gate import (R1_SPLIT_PURPOSE, TrainingAdmissionError,
                                          admit_full_training)


def ready_pilot():
    return {
        "status": READY, "termination_cause": None, "gradient_finite": True,
        "pilot_target_steps": PILOT_STEPS, "completed_steps": PILOT_STEPS,
        "protocol_metadata": expected_pilot_protocol_metadata(),
        "median_seconds_per_step": 0.25, "setup_overhead_seconds": 10.0,
        "validation_overhead_seconds": 20.0,
        "peak_host_rss_bytes": 1.0, "peak_gpu_allocated_bytes": 2.0,
        "peak_gpu_reserved_bytes": 3.0,
    }


class TrainingGateTests(unittest.TestCase):
    def test_exact_budget_eta_and_fixed_identity(self):
        plan = admit_full_training(train_loader=range(3), pilot=ready_pilot(), split_purpose=R1_SPLIT_PURPOSE)
        self.assertEqual(plan["max_steps"], 3_000)
        self.assertEqual(plan["eta_seconds"], 780.0)
        self.assertEqual(plan["protocol_metadata"]["precision"], "32-true")
        self.assertEqual(plan["split_purpose"], dict(R1_SPLIT_PURPOSE))

    def test_absent_or_unready_pilot_stops(self):
        with self.assertRaises(TrainingAdmissionError):
            admit_full_training(train_loader=range(1), pilot={}, split_purpose=R1_SPLIT_PURPOSE)
        pilot = ready_pilot(); pilot["status"] = "STOPPED_INCOMPLETE"
        with self.assertRaises(TrainingAdmissionError):
            admit_full_training(train_loader=range(1), pilot=pilot, split_purpose=R1_SPLIT_PURPOSE)

    def test_oom_nan_and_forged_provenance_stop(self):
        for field, value in (("termination_cause", "OOM"), ("median_seconds_per_step", float("nan"))):
            pilot = ready_pilot(); pilot[field] = value
            with self.subTest(field=field), self.assertRaises(TrainingAdmissionError):
                admit_full_training(train_loader=range(1), pilot=pilot, split_purpose=R1_SPLIT_PURPOSE)
        pilot = ready_pilot(); pilot["protocol_metadata"] = {"precision": "fake"}
        with self.assertRaises(TrainingAdmissionError):
            admit_full_training(train_loader=range(1), pilot=pilot, split_purpose=R1_SPLIT_PURPOSE)

    def test_test_access_and_invalid_loader_are_forbidden(self):
        purpose = dict(R1_SPLIT_PURPOSE); purpose["test"] = "evaluation"
        with self.assertRaises(TrainingAdmissionError):
            admit_full_training(train_loader=range(1), pilot=ready_pilot(), split_purpose=purpose)
        with self.assertRaises(ValueError):
            admit_full_training(train_loader=range(0), pilot=ready_pilot(), split_purpose=R1_SPLIT_PURPOSE)


if __name__ == "__main__":
    unittest.main()
