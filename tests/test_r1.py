import math
import unittest

from fine_defect_ad.r1 import (
    LOW_FPR_CLAIM_STATE, audit_testpub_normal, calibrate_raw_threshold,
    clopper_pearson_upper, freeze_r1_contract, raw_image_score,
    threshold_raw_map, validate_efficientad_s_config,
)


class R1Tests(unittest.TestCase):
    def config(self):
        return {"image_size": (256, 256), "batch_size": 1, "model_size": "small", "learning_rate": 1e-4,
                "weight_decay": 1e-5, "max_steps": 70000, "max_epochs": 1000, "normalization": None,
                "seeds": (7, 11), "pilot_steps": 1000}

    def test_fixed_config_and_no_test_derived_fields(self):
        self.assertEqual(validate_efficientad_s_config(self.config()).max_steps, 70000)
        for key, value in (("batch_size", 2), ("normalization", "Normalize"), ("target_fpr", .01)):
            bad = self.config(); bad[key] = value
            with self.assertRaises(ValueError): validate_efficientad_s_config(bad)

    def test_raw_threshold_requires_provenance_and_is_shared(self):
        with self.assertRaisesRegex(ValueError, "provenance"): calibrate_raw_threshold([[1]], {})
        threshold = calibrate_raw_threshold([[1, 2], [3, 4]], {"comparator": ">"})
        self.assertAlmostEqual(threshold.value, 2.5 + 3 * math.sqrt(1.25))
        raw = [[1, 7]]; preserved, image_positive, score = threshold_raw_map(raw, threshold)
        self.assertIs(preserved, raw); self.assertEqual(score, raw_image_score(raw)); self.assertTrue(image_positive)

    def test_exact_bounds_and_conservative_public_audit(self):
        self.assertAlmostEqual(clopper_pearson_upper(0, 10, .95), 1 - .05 ** .1)
        self.assertEqual(clopper_pearson_upper(10, 10, .95), 1.0)
        self.assertGreater(clopper_pearson_upper(1, 10, .95), .1)
        threshold = calibrate_raw_threshold([[0]], {"comparator": ">"})
        for maps, positives in (([[0]], 0), ([[1]], 1), ([[1], [1]], 2)):
            audit = audit_testpub_normal(maps, threshold, confidence=.95, minimum_normal_samples=24)
            self.assertEqual((audit.normal_count, audit.positives), (len(maps), positives))
            self.assertFalse(audit.sample_size_sufficient)
            self.assertEqual(audit.claim_state, LOW_FPR_CLAIM_STATE)
        with self.assertRaises(ValueError): audit_testpub_normal([], threshold, confidence=.95, minimum_normal_samples=24)

    def test_contract_freezes_config_and_excludes_results(self):
        config = self.config(); contract = freeze_r1_contract(config, ["pub-2", "pub-1"])
        config["batch_size"] = 2
        self.assertEqual(contract.config_hash, freeze_r1_contract(self.config(), ["pub-1", "pub-2"]).config_hash)
        self.assertEqual(contract.test_identity_hash, freeze_r1_contract(self.config(), ["pub-2", "pub-1"]).test_identity_hash)
        with self.assertRaisesRegex(ValueError, "unique"): freeze_r1_contract(self.config(), ["pub-1", "pub-1"])


if __name__ == "__main__":
    unittest.main()
