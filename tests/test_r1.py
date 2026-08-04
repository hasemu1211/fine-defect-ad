import json
import math
from pathlib import Path
import unittest

from fine_defect_ad.r1 import (
    LOW_FPR_CLAIM_STATE, PROTOCOL_BLOCKED_STATE, R1_SEED, R1_SEED_DERIVATION,
    R1_SEED_IDENTITY, R1_SEED_IDENTITY_SHA256, audit_testpub_normal,
    calibrate_raw_threshold, clopper_pearson_upper, freeze_r1_contract,
    RawThreshold, raw_image_score, threshold_raw_map,
    validate_efficientad_s_config,
)


SELF_ASSERTED_COMPARATOR = {"formula_status": "VERIFIED", "formula_source_locator": "MVTec AD 2 paper §4.2.2",
                            "comparator_status": "VERIFIED", "comparator": ">",
                            "source_locator": "https://example.invalid/protocol", "source_sha256": "0" * 64}
SEED_PROVENANCE = {"status": "VERIFIED", "upstream_seed_status": "ABSENT",
                   "identity": R1_SEED_IDENTITY, "identity_sha256": R1_SEED_IDENTITY_SHA256,
                   "derivation": R1_SEED_DERIVATION, "seed": R1_SEED}


class R1Tests(unittest.TestCase):
    def config(self):
        return {"image_size": (256, 256), "batch_size": 1, "model_size": "small", "learning_rate": 1e-4,
                "weight_decay": 1e-5, "max_steps": 70000, "max_epochs": 1000, "normalization": None,
                "seeds": (R1_SEED,), "seed_provenance": SEED_PROVENANCE, "pilot_steps": 1000}

    def test_fixed_config_requires_verified_seed_provenance(self):
        self.assertEqual(json.loads(Path("evidence/r1-seed-provenance.json").read_text())["seed"], R1_SEED)
        self.assertEqual(validate_efficientad_s_config(self.config()).max_steps, 70000)
        for key, value in (("batch_size", 2), ("normalization", "Normalize"), ("target_fpr", .01),
                           ("seed_provenance", {})):
            bad = self.config(); bad[key] = value
            with self.assertRaises(ValueError): validate_efficientad_s_config(bad)

    def test_numeric_threshold_stays_blocked_even_with_self_asserted_comparator(self):
        evidence = json.loads(Path("evidence/mvtec-metric-protocol.json").read_text())
        self.assertEqual(evidence["official_benchmark_claim"], PROTOCOL_BLOCKED_STATE)
        threshold = calibrate_raw_threshold([[1, 2], [3, 4]], SELF_ASSERTED_COMPARATOR)
        self.assertAlmostEqual(threshold.value, 2.5 + 3 * math.sqrt(1.25))
        self.assertIsNone(threshold.comparator)
        with self.assertRaisesRegex(ValueError, "blocked"): threshold_raw_map([[1]], threshold)
        with self.assertRaisesRegex(ValueError, "blocked"): audit_testpub_normal([[[1]]], threshold, confidence=.95, minimum_normal_samples=1)

    def test_direct_comparator_construction_cannot_bypass_fail_closed_gate(self):
        for comparator in (">", ">="):
            threshold = RawThreshold(1.0, comparator, SELF_ASSERTED_COMPARATOR)
            with self.assertRaisesRegex(ValueError, "blocked"):
                threshold.is_positive(2.0)
            with self.assertRaisesRegex(ValueError, "blocked"):
                threshold_raw_map([[2.0]], threshold)

    def test_exact_bounds_and_conservative_public_audit(self):
        self.assertAlmostEqual(clopper_pearson_upper(0, 10, .95), 1 - .05 ** .1)
        self.assertEqual(clopper_pearson_upper(10, 10, .95), 1.0)
        self.assertGreater(clopper_pearson_upper(1, 10, .95), .1)
        threshold = calibrate_raw_threshold([[0]], SELF_ASSERTED_COMPARATOR)
        with self.assertRaisesRegex(ValueError, "blocked"):
            audit_testpub_normal([[0]], threshold, confidence=.95, minimum_normal_samples=24)

    def test_contract_freezes_config_and_excludes_results(self):
        config = self.config(); contract = freeze_r1_contract(config, ["pub-2", "pub-1"])
        config["batch_size"] = 2
        self.assertEqual(contract.config_hash, freeze_r1_contract(self.config(), ["pub-1", "pub-2"]).config_hash)
        self.assertEqual(contract.test_identity_hash, freeze_r1_contract(self.config(), ["pub-2", "pub-1"]).test_identity_hash)
        with self.assertRaisesRegex(ValueError, "unique"): freeze_r1_contract(self.config(), ["pub-1", "pub-1"])


if __name__ == "__main__":
    unittest.main()
