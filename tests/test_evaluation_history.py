import csv
import json
from hashlib import sha256

import numpy as np

from fine_defect_ad.evaluation_history import build_evaluation_history


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
    baseline = write_json("baseline.json", {"local_au_pro": {"output": {"score": .2}}, "command": "base", "protocol": "frozen"})
    candidate_binding = write_json("binding.json", {"checkpoint_sha256": checkpoint})
    candidate = write_json("candidate.json", {"local_au_pro": {"output": {"score": .5}}, "command": "candidate", "protocol": "frozen",
                                                  "split_freeze_sha256": sha256(candidate_binding.read_bytes()).hexdigest()})
    base_manifest = manifest("base-maps", [("normal", [0., 1.]), ("anomaly", [0., .2])])
    candidate_manifest = manifest("candidate-maps", [("normal", [0., .5]), ("anomaly", [0., .8])])
    base_value=json.loads(base_manifest.read_text())
    for row in base_value["maps"]: row["checkpoint_sha256"] = checkpoint
    base_manifest.write_text(json.dumps(base_value))
    result = build_evaluation_history(baseline_evidence=baseline, candidate_evidence=candidate,
                                      baseline_manifest=base_manifest, candidate_manifest=candidate_manifest,
                                      candidate_binding=candidate_binding, artifact_root=tmp_path, output_dir=tmp_path / "out",
                                      metric_path="local_au_pro.output.score", change_scope="pipeline_only",
                                      change_description="same checkpoint", normal_label="normal", anomaly_label="anomaly",
                                      baseline_label="baseline", candidate_label="candidate")
    comparison = json.loads(result["comparison"].read_text())
    assert comparison["absolute_change"] == .3 and comparison["change_scope"] == "pipeline_only"
    rows = list(csv.DictReader(result["error_cases"].open()))
    assert len(rows) == 2 and {row["classification_status"] for row in rows} == {"NOT_COMPUTED_NO_FROZEN_THRESHOLD"}
    assert comparison["verified_bindings"]["checkpoint_sha256"] == checkpoint
    assert result["metric_visual"].read_text().startswith("<svg") and result["review_visual"].read_text().startswith("<svg")
