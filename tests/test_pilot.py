import json
import unittest

from fine_defect_ad.pilot import (PILOT_STEPS, READY, STOPPED_INCOMPLETE, PilotEvidence,
                                  estimate_eta_seconds, pilot_step_budget)


class PilotTests(unittest.TestCase):
    def test_budget_and_eta_use_only_measurements(self):
        self.assertEqual(pilot_step_budget(range(100)), 70_000)
        self.assertEqual(pilot_step_budget(range(3)), 3_000)
        self.assertEqual(estimate_eta_seconds(total_steps=10, step_timestamps=[1, 3, 8],
                                               setup_overhead_seconds=2, validation_overhead_seconds=4), 41)
        with self.assertRaises(ValueError):
            estimate_eta_seconds(total_steps=10, step_timestamps=[1, 2], setup_overhead_seconds=None,
                                 validation_overhead_seconds=1)

    def test_nan_or_incomplete_pilot_stops(self):
        evidence = PilotEvidence("r", "runner", 3_000); evidence.record_setup(1); evidence.record_validation(2)
        evidence.record_step(timestamp=1, gradients_finite=False, host_rss_bytes=10)
        record = evidence.to_record()
        self.assertEqual(record["status"], STOPPED_INCOMPLETE)
        self.assertEqual(record["termination_cause"], "GRADIENT_NONFINITE")

    def test_exact_1000_is_ready_and_json_compatible(self):
        evidence = PilotEvidence("r", "runner", 3_000); evidence.record_setup(1); evidence.record_validation(2)
        for timestamp in range(PILOT_STEPS):
            evidence.record_step(timestamp=timestamp, gradients_finite=True, host_rss_bytes=timestamp,
                                 gpu_allocated_bytes=timestamp * 2, gpu_reserved_bytes=timestamp * 3)
        record = evidence.to_record()
        self.assertEqual(record["status"], READY)
        self.assertEqual(record["completed_steps"], PILOT_STEPS)
        self.assertEqual(record["peak_gpu_reserved_bytes"], 2997.0)
        json.dumps(record, allow_nan=False)


class PilotRunnerTests(unittest.TestCase):
    def test_runner_holds_one_lease_for_setup_steps_and_validation(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from fine_defect_ad.pilot import run_pilot

        ticks = iter(range(2_000))
        with TemporaryDirectory() as raw:
            record = run_pilot(lease_directory=Path(raw), run_id="lease-run", command="pilot",
                               train_loader=range(1), setup=lambda: None,
                               step=lambda: {"gradients_finite": True, "gpu_allocated_bytes": 1,
                                             "gpu_reserved_bytes": 2}, validate=lambda: None,
                               clock=lambda: float(next(ticks)))
        self.assertEqual(record["status"], READY)
        self.assertEqual([event["state"] for event in record["lease_events"]], ["acquired", "released"])
        self.assertTrue(all(event["run_id"] == "lease-run" for event in record["lease_events"]))


    def test_nan_skips_validation_but_releases_lease(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from fine_defect_ad.pilot import run_pilot

        validated = []
        ticks = iter(range(10))
        with TemporaryDirectory() as raw:
            record = run_pilot(lease_directory=Path(raw), run_id="nan-run", command="pilot",
                               train_loader=range(1), setup=lambda: None,
                               step=lambda: {"gradients_finite": False},
                               validate=lambda: validated.append(True),
                               clock=lambda: float(next(ticks)))
        self.assertEqual(validated, [])
        self.assertEqual(record["status"], STOPPED_INCOMPLETE)
        self.assertEqual(record["termination_cause"], "GRADIENT_NONFINITE")
        self.assertIsNone(record["validation_overhead_seconds"])
        self.assertEqual(record["lease_events"][-1]["state"], "released")


if __name__ == "__main__":
    unittest.main()
