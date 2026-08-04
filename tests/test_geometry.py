import unittest

import numpy as np

from fine_defect_ad.geometry import (
    NO_EXTERNAL_MINIMUM_AVAILABLE, BorderEvidence, derive_tile_plan,
    geometry_candidates, stitch_tiles, valid_region,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.border = BorderEvidence(1, 1, 1, 1, "synthetic-border-diagnostic")

    def test_geo_001_tiles_cover_without_oob_or_gaps(self):
        for source, model in (((7, 9), (4, 5)), ((4, 5), (4, 5)), ((11, 6), (4, 4))):
            plan = derive_tile_plan(source, model, self.border)
            coverage = np.zeros(source, dtype=int)
            for tile in plan.tiles:
                self.assertTrue(0 <= tile.y0 < tile.y1 <= source[0])
                self.assertTrue(0 <= tile.x0 < tile.x1 <= source[1])
                region = valid_region(tile, plan)
                coverage[region.y0:region.y1, region.x0:region.x1] += 1
            self.assertTrue(np.all(coverage > 0))
        with self.assertRaisesRegex(ValueError, "overlap"):
            derive_tile_plan((7, 9), (4, 5), self.border, overlap=0.5)

    def test_geo_002_ramp_stitches_identity(self):
        source = np.add.outer(np.arange(8) * 100, np.arange(9)).astype(float)
        plan = derive_tile_plan(source.shape, (4, 5), self.border)
        maps = [source[t.y0:t.y1, t.x0:t.x1] for t in plan.tiles]
        np.testing.assert_array_equal(stitch_tiles(plan, maps), source)

    def test_geo_003_impulse_and_lines_preserve_across_seams(self):
        source = np.zeros((9, 10), dtype=float)
        source[4, :] = 1
        source[:, 5] = 2
        source[4, 5] = 3
        plan = derive_tile_plan(source.shape, (4, 5), self.border)
        maps = [source[t.y0:t.y1, t.x0:t.x1] for t in plan.tiles]
        np.testing.assert_array_equal(stitch_tiles(plan, maps), source)

    def test_geo_004_no_external_scale_has_one_exploratory_e2(self):
        candidates = geometry_candidates((8, 9), (4, 5), self.border)
        self.assertEqual([candidate.identifier for candidate in candidates], ["E1", "E2"])
        self.assertEqual(candidates[0].external_minimum_status, NO_EXTERNAL_MINIMUM_AVAILABLE)
        self.assertEqual(candidates[1].external_minimum_status, NO_EXTERNAL_MINIMUM_AVAILABLE)
        self.assertEqual(candidates[1].kind, "tiled_exploratory")


if __name__ == "__main__":
    unittest.main()
