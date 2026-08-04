import json
import re
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_geometry_evidence_is_candidate_only_and_public_safe():
    evidence = json.loads(Path("evidence/g002-geometry-decision.json").read_text())

    assert evidence["decision_id"] == "DEC-GEO-002"
    assert evidence["status"] == "CANDIDATE_VALIDATION_REQUIRED"
    local = evidence["local_pdn_derivation"]
    assert (local["receptive_field_pixels"], local["effective_stride_pixels"]) == (33, 4)
    assert (local["input_256_unpadded_feature_map_cells"], local["pad_maps_cells_per_edge"]) == (56, 4)
    candidate = evidence["candidate_tiling_geometry"]
    assert candidate["status"] == "CANDIDATE_DERIVED_FROM_LOCAL_PDN_ONLY"
    assert (candidate["local_border_pixels"], candidate["overlap_pixels"]) == (16, 32)
    limitation = evidence["autoencoder_limitation"]
    assert limitation["encoder_receptive_field_pixels"] == 318
    assert limitation["input_256_latent_shape"] == "1x1"
    assert limitation["combined_map_has_tile_global_context"] is True
    assert limitation["overlap_seam_equivalence_status"] == "NOT_GUARANTEED"
    gate = evidence["finalization_gate"]
    assert gate["selection_data_scope"] == "NORMAL_VALIDATION_ONLY"
    assert set(gate["forbidden_selection_data"]) == {"TESTpub", "TESTpriv", "TESTpriv,mix", "OOD"}
    assert gate["external_minimum_defect_status"] == "NO_EXTERNAL_MINIMUM_AVAILABLE"
    assert gate["production_defect_size_claim"] == "BLOCKED"
    sources = evidence["primary_sources"]
    assert sources["efficientad_wacv_2024"]["url"].startswith("https://")
    assert HEX64.fullmatch(sources["efficientad_wacv_2024"]["sha256"])
    pinned = sources["anomalib_2_6_0_pinned"]
    assert len(pinned["commit"]) == 40
    assert all(HEX64.fullmatch(pinned[key]) for key in ("torch_model_sha256", "lightning_model_sha256"))
    assert all(not value.startswith(("/", "~/")) for value in _strings(evidence))
