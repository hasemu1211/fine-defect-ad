import csv
import json
from hashlib import sha256

import numpy as np
import pytest

from fine_defect_ad.evaluation_history import _image_auroc, build_evaluation_history
from fine_defect_ad.g002_e2_runtime import _canonical


def test_image_auroc_is_tie_aware_and_rejects_nonbinary_labels():
    tied = {"good": {"label": "good", "max": .5}, "bad": {"label": "bad", "max": .5}}
    assert _image_auroc(tied, normal_label="good", anomaly_label="bad") == .5
    tied["bad"]["label"] = "unknown"
    try:
        _image_auroc(tied, normal_label="good", anomaly_label="bad")
    except ValueError as exc:
        assert "non-binary" in str(exc)
    else:
        raise AssertionError("non-binary labels must fail closed")


def test_build_evaluation_history_is_input_driven(tmp_path):
    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    def manifest(name, values):
        rows = []
        for index, (label, data) in enumerate(values):
            raw = np.asarray(data, dtype="<f4").tobytes(); digest = sha256(raw).hexdigest()
            (tmp_path / f"{name}-{index}-{digest}.bin").write_bytes(raw)
            rows.append({"image_identity": f"test/{label}/{index}.png", "label": label,
                         "source_sha256": str(index) * 64, "map_sha256": digest, "shape": [len(data)]})
        return write_json(f"{name}.json", {"maps": rows})
    checkpoint = "c" * 64
    lineage = {key: checkpoint for key in ("checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256")}
    binding_value = {**lineage, "status": "READY", "stage": "PRE_TEST_FREEZE", "decision_id": "DEC-SPLIT-003",
                     "protocol": "SPLIT_ST_SOURCE_RESOLUTION__GLOBAL_STAE_CANONICAL_256__NO_TEST_ACCESS",
                     "validation_identities": [], "geometry": {}, "quantiles": {"qa_st": 0., "qb_st": 1., "qa_stae": 0., "qb_stae": 1.},
                     "maps": [], "code_sha256": checkpoint}
    binding_value["freeze_sha256"] = sha256(_canonical(binding_value)).hexdigest()
    candidate_binding = write_json("binding.json", binding_value)
    base_manifest = manifest("base-maps", [("normal", [0., 1.]), ("anomaly", [0., .2])])
    candidate_manifest = manifest("candidate-maps", [("normal", [0., .5]), ("anomaly", [0., .8])])
    base_value=json.loads(base_manifest.read_text())
    base_value.update({"status": "TEST_PUBLIC_RAW_MAPS_ONLY", "selected_measurement": "E1", "lineage": lineage})
    for row in base_value["maps"]: row["checkpoint_sha256"] = checkpoint
    base_manifest.write_text(json.dumps(base_value))
    candidate_value = json.loads(candidate_manifest.read_text())
    candidate_value.update({"status": "SPLIT_E2_TEST_PUBLIC_RAW_MAPS", "freeze_sha256": binding_value["freeze_sha256"], "decision_id": "DEC-SPLIT-003"})
    candidate_manifest.write_text(json.dumps(candidate_value))
    baseline = write_json("baseline.json", {"local_au_pro": {"output": {"score": .2}}, "command": "base", "status": "READY",
                                              "protocol": "TEST_PUBLIC_RAW_MAPS_ONLY_NO_SELECTION_OR_CALIBRATION_MUTATION", "selected_measurement": "E1",
                                              "lineage": lineage, "raw_manifest_sha256": sha256(base_manifest.read_bytes()).hexdigest()})
    candidate = write_json("candidate.json", {"local_au_pro": {"output": {"score": .5}}, "command": "candidate", "status": "READY",
                                                "protocol": "POST_HOC_PIPELINE_CORRECTION__ONE_SHOT_TESTPUB__NO_TUNING", "decision_id": "DEC-SPLIT-003",
                                                "raw_manifest_sha256": sha256(candidate_manifest.read_bytes()).hexdigest(),
                                                "split_freeze_sha256": sha256(candidate_binding.read_bytes()).hexdigest()})
    result = build_evaluation_history(baseline_evidence=baseline, candidate_evidence=candidate,
                                      baseline_manifest=base_manifest, candidate_manifest=candidate_manifest,
                                      candidate_binding=candidate_binding, artifact_root=tmp_path, output_dir=tmp_path / "out",
                                      metric_path="local_au_pro.output.score", change_scope="pipeline_only",
                                      change_description="same checkpoint", normal_label="normal", anomaly_label="anomaly",
                                      baseline_label="base & <one>", candidate_label="candidate & <two>")
    comparison = json.loads(result["comparison"].read_text())
    assert comparison["absolute_change"] == .3 and comparison["change_scope"] == "pipeline_only"
    assert comparison["metrics"]["image_auroc"]["baseline"] == 0.0
    assert comparison["metrics"]["image_auroc"]["candidate"] == 1.0
    rows = list(csv.DictReader(result["error_cases"].open()))
    assert len(rows) == 2 and {row["classification_status"] for row in rows} == {"NOT_COMPUTED_NO_FROZEN_THRESHOLD"}
    assert comparison["verified_bindings"]["checkpoint_sha256"] == checkpoint
    assert result["metric_visual"].read_text().startswith("<svg") and result["review_visual"].read_text().startswith("<svg")
    visual = result["metric_visual"].read_text()
    assert "base &amp; &lt;one&gt;" in visual and "candidate &amp; &lt;two&gt;" in visual

    for document, key, replacement in ((baseline, "raw_manifest_sha256", "x" * 64),
                                       (candidate, "split_freeze_sha256", "x" * 64),
                                       (candidate, "status", "BLOCKED"),
                                       (candidate, "protocol", "wrong"),
                                       (candidate_manifest, "status", "wrong"),
                                       (candidate_binding, "protocol", "wrong")):
        original = json.loads(document.read_text()); altered = {**original, key: replacement}; document.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            build_evaluation_history(baseline_evidence=baseline, candidate_evidence=candidate,
                                     baseline_manifest=base_manifest, candidate_manifest=candidate_manifest,
                                     candidate_binding=candidate_binding, artifact_root=tmp_path, output_dir=tmp_path / "blocked",
                                     metric_path="local_au_pro.output.score", change_scope="pipeline_only",
                                     change_description="same checkpoint", normal_label="normal", anomaly_label="anomaly",
                                     baseline_label="base & <one>", candidate_label="candidate & <two>")
        document.write_text(json.dumps(original))

    def reject_tampered_freeze(value):
        candidate_binding.write_text(json.dumps(value))
        candidate_value = json.loads(candidate.read_text())
        candidate_value["split_freeze_sha256"] = sha256(candidate_binding.read_bytes()).hexdigest()
        candidate.write_text(json.dumps(candidate_value))
        with pytest.raises(ValueError):
            build_evaluation_history(baseline_evidence=baseline, candidate_evidence=candidate,
                                     baseline_manifest=base_manifest, candidate_manifest=candidate_manifest,
                                     candidate_binding=candidate_binding, artifact_root=tmp_path, output_dir=tmp_path / "blocked",
                                     metric_path="local_au_pro.output.score", change_scope="pipeline_only",
                                     change_description="same checkpoint", normal_label="normal", anomaly_label="anomaly",
                                     baseline_label="base", candidate_label="candidate")

    changed = dict(binding_value); changed["code_sha256"] = "x" * 64
    reject_tampered_freeze(changed)  # Valid external file hash, invalid canonical freeze self-hash.
    changed = dict(binding_value); changed.pop("quantiles")
    reject_tampered_freeze(changed)  # Required freeze structure must remain intact.
