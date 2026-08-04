from pathlib import Path
import unittest


class R1OverlayLockTests(unittest.TestCase):
    def test_overlay_lock_pins_anomalib_without_base_cuda_stack(self):
        lock = Path("requirements/r1-overlay.txt").read_text().lower()
        self.assertIn("anomalib @ https://files.pythonhosted.org/", lock)
        self.assertIn("--hash=sha256:0395d2e2ad859fb45b9c4544479639afe5d6aaada5e2aefc460bb65b638bd972", lock)
        packages = "\n".join(line for line in lock.splitlines() if not line.startswith("#"))
        for forbidden in ("torch @", "torchvision @", "nvidia-", "cuda"):
            self.assertNotIn(forbidden, packages)


if __name__ == "__main__":
    unittest.main()
