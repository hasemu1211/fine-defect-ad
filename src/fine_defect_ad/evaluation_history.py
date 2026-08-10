"""Build reusable baseline, comparison, and review-priority evaluation records."""
from __future__ import annotations

import argparse
import csv
import json
from html import escape
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .g002_e2_runtime import verify_split_freeze


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _lookup(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"metric path not found: {dotted_path}")
        current = current[key]
    return current


def _map_stats(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    rows = manifest.get("maps")
    if not isinstance(rows, list) or not rows:
        raise ValueError("non-empty maps list required")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("map row must be an object")
        digest, identity, shape = row.get("map_sha256"), row.get("image_identity"), row.get("shape")
        matches = list(root.glob(f"*{digest}.bin")) if isinstance(digest, str) else []
        if len(matches) != 1 or not isinstance(identity, str) or not isinstance(shape, list):
            raise ValueError(f"map artifact lookup failed: {identity}")
        if identity in result:
            raise ValueError(f"duplicate map identity: {identity}")
        raw = matches[0].read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"map hash mismatch: {identity}")
        data = np.frombuffer(raw, dtype="<f4")
        if data.size != int(np.prod(shape)) or not bool(np.isfinite(data).all()):
            raise ValueError(f"invalid map payload: {identity}")
        result[identity] = {
            "label": row.get("label"),
            "source_sha256": row.get("source_sha256"),
            "mean": float(data.mean(dtype=np.float64)),
            "p99": float(np.quantile(data, .99)),
            "max": float(data.max()),
        }
    return result


def _identity_sha256(stats: dict[str, dict[str, Any]]) -> str:
    rows = [[identity, value["label"], value["source_sha256"]] for identity, value in sorted(stats.items())]
    return sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _image_auroc(stats: dict[str, dict[str, Any]], *, normal_label: str, anomaly_label: str) -> float:
    """Compute tie-aware binary AUROC from each image's maximum raw-map score."""
    rows: list[tuple[float, int]] = []
    for identity, value in stats.items():
        label = value.get("label")
        if label == normal_label:
            target = 0
        elif label == anomaly_label:
            target = 1
        else:
            raise ValueError(f"non-binary image label: {identity}")
        score = value.get("max")
        if not isinstance(score, (int, float)) or not bool(np.isfinite(score)):
            raise ValueError(f"non-finite image score: {identity}")
        rows.append((float(score), target))
    positives = sum(target for _, target in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError("both normal and anomaly images required for AUROC")
    rows.sort(key=lambda row: row[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end][0] == rows[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(target for _, target in rows[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _verify_input_bindings(*, baseline: dict[str, Any], candidate: dict[str, Any], base_maps: dict[str, Any],
                           candidate_maps: dict[str, Any], binding: dict[str, Any], baseline_manifest: Path,
                           candidate_manifest: Path, candidate_binding: Path) -> str:
    """Reject evidence/manifests unless their persisted production bindings agree."""
    base_hash, candidate_hash, binding_hash = _hash(baseline_manifest), _hash(candidate_manifest), _hash(candidate_binding)
    if baseline.get("raw_manifest_sha256") != base_hash or candidate.get("raw_manifest_sha256") != candidate_hash:
        raise ValueError("evidence/raw manifest hash binding mismatch")
    if candidate.get("split_freeze_sha256") != binding_hash:
        raise ValueError("candidate evidence/split freeze hash binding mismatch")
    verify_split_freeze(binding)
    if (baseline.get("status"), baseline.get("protocol"), baseline.get("selected_measurement")) != (
        "READY", "TEST_PUBLIC_RAW_MAPS_ONLY_NO_SELECTION_OR_CALIBRATION_MUTATION", "E1"):
        raise ValueError("baseline evidence status/protocol mismatch")
    if (base_maps.get("status"), base_maps.get("selected_measurement")) != ("TEST_PUBLIC_RAW_MAPS_ONLY", "E1"):
        raise ValueError("baseline manifest status/protocol mismatch")
    if (candidate.get("status"), candidate.get("protocol")) != (
        "READY", "POST_HOC_PIPELINE_CORRECTION__ONE_SHOT_TESTPUB__NO_TUNING"):
        raise ValueError("candidate evidence status/protocol mismatch")
    if candidate_maps.get("status") != "SPLIT_E2_TEST_PUBLIC_RAW_MAPS":
        raise ValueError("candidate manifest status/protocol mismatch")
    if (binding.get("status"), binding.get("stage"), binding.get("protocol")) != (
        "READY", "PRE_TEST_FREEZE", "SPLIT_ST_SOURCE_RESOLUTION__GLOBAL_STAE_CANONICAL_256__NO_TEST_ACCESS"):
        raise ValueError("candidate freeze status/protocol mismatch")
    if candidate_maps.get("freeze_sha256") != binding.get("freeze_sha256"):
        raise ValueError("candidate manifest/freeze identifier mismatch")
    if candidate.get("decision_id") != binding.get("decision_id") or candidate_maps.get("decision_id") != binding.get("decision_id"):
        raise ValueError("candidate decision binding mismatch")
    lineage = baseline.get("lineage")
    manifest_lineage = base_maps.get("lineage")
    if not isinstance(lineage, dict) or not isinstance(manifest_lineage, dict):
        raise ValueError("baseline lineage missing")
    keys = ("checkpoint_sha256", "sidecar_sha256", "metrics_sha256", "final_attempt_sha256", "identity_sha256", "pilot_sha256")
    for key in keys:
        if lineage.get(key) != manifest_lineage.get(key) or lineage.get(key) != binding.get(key):
            raise ValueError(f"lineage binding mismatch: {key}")
    checkpoint = lineage["checkpoint_sha256"]
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint lineage missing")
    return checkpoint


def _write_visuals(output_dir: Path, comparison: dict[str, Any], rows: list[dict[str, Any]],
                   baseline_label: str, candidate_label: str) -> tuple[Path, Path]:
    localization, detection = comparison["metrics"]["localization_au_pro_0_05"], comparison["metrics"]["image_auroc"]
    def metric_row(y: int, title: str, detail: str, values: dict[str, Any]) -> str:
        return (f'<text x="94" y="{y}" class="metric">{title}</text><text x="94" y="{y+23}" class="sub">{detail}</text>'
                f'<text x="690" y="{y}" class="label" text-anchor="middle">{escape(baseline_label)}</text><text x="690" y="{y+27}" class="value" text-anchor="middle">{values["baseline"]:.6f}</text>'
                f'<text x="930" y="{y}" class="label" text-anchor="middle">{escape(candidate_label)}</text><text x="930" y="{y+27}" class="value" text-anchor="middle">{values["candidate"]:.6f}</text>')
    metric_svg = output_dir / "metric-comparison.svg"
    metric_svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="360" viewBox="0 0 1120 360">'
                          '<style>text{font-family:system-ui,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#526070}.metric{font-size:16px;font-weight:700}.label{font-size:12px;font-weight:600;fill:#526070}.value{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}</style>'
                          '<rect width="100%" height="100%" fill="#f8fafc"/><text x="560" y="48" class="title" text-anchor="middle">TESTpub 평가 비교</text>'
                          '<text x="560" y="75" class="sub" text-anchor="middle">같은 입력·체크포인트 · 지표별 수치를 독립 표기</text>'
                          '<line x1="58" y1="105" x2="1062" y2="105" stroke="#cbd5e1"/>'
                          f'{metric_row(143, "Image AU-ROC", "이미지 단위 이상 순위 · raw map 최대값", detection)}'
                          '<line x1="58" y1="211" x2="1062" y2="211" stroke="#cbd5e1"/>'
                          f'{metric_row(249, "AU-PRO@0.05", "픽셀 위치화 · local evaluator", localization)}'
                          '<text x="560" y="329" class="sub" text-anchor="middle">상대 막대는 서로 다른 지표의 크기를 혼동시킬 수 있어 사용하지 않았습니다.</text></svg>\n', encoding="utf-8")

    selected = sorted(rows, key=lambda row: (row["label"], row["review_rank"]))
    selected = [row for row in selected if row["review_rank"] <= 5]
    review_body = []
    for index, row in enumerate(selected):
        y = 94 + index * 24
        review_body.append(f'<text x="36" y="{y}" class="row">{escape(str(row["label"]))} #{row["review_rank"]}</text>'
                           f'<text x="145" y="{y}" class="row">{escape(str(row["image_identity"])[-42:])}</text>'
                           f'<text x="610" y="{y}" class="score">p99 {row["candidate_p99"]:.5f}</text>')
    review_svg = output_dir / "review-priority.svg"
    review_svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">'
                          '<style>text{font-family:system-ui,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#526070}.row{font-size:12px}.score{font-size:12px;font-variant-numeric:tabular-nums}</style>'
                          '<rect width="100%" height="100%" fill="#f8fafc"/><text x="36" y="42" class="title">오류 검토 우선순위</text>'
                          '<text x="36" y="66" class="sub">정상 고점수·불량 저점수 각 상위 5건 · 판정 임계값 미적용</text>'
                          f'{"".join(review_body)}</svg>\n', encoding="utf-8")
    return metric_svg, review_svg


def build_evaluation_history(*, baseline_evidence: Path, candidate_evidence: Path,
                             baseline_manifest: Path, candidate_manifest: Path,
                             candidate_binding: Path, artifact_root: Path, output_dir: Path, metric_path: str,
                             change_scope: str, change_description: str, normal_label: str,
                             anomaly_label: str, baseline_label: str, candidate_label: str) -> dict[str, Path]:
    paths = [baseline_evidence, candidate_evidence, baseline_manifest, candidate_manifest]
    baseline, candidate, base_maps, candidate_maps = map(_read, paths)
    baseline_value, candidate_value = float(_lookup(baseline, metric_path)), float(_lookup(candidate, metric_path))
    if not bool(np.isfinite((baseline_value, candidate_value)).all()):
        raise ValueError("metrics must be finite")
    if normal_label == anomaly_label:
        raise ValueError("normal/anomaly labels must differ")
    base_stats, candidate_stats = _map_stats(artifact_root, base_maps), _map_stats(artifact_root, candidate_maps)
    if base_stats.keys() != candidate_stats.keys():
        raise ValueError("baseline/candidate identities differ")
    for identity in base_stats:
        if any(base_stats[identity][key] != candidate_stats[identity][key] for key in ("label", "source_sha256")):
            raise ValueError(f"baseline/candidate identity binding differs: {identity}")
    labels = {value["label"] for value in base_stats.values()}
    if labels != {normal_label, anomaly_label}:
        raise ValueError(f"label mapping does not cover manifest labels: {sorted(labels)}")
    binding = _read(candidate_binding)
    checkpoint_sha256 = _verify_input_bindings(baseline=baseline, candidate=candidate, base_maps=base_maps,
                                                candidate_maps=candidate_maps, binding=binding,
                                                baseline_manifest=baseline_manifest, candidate_manifest=candidate_manifest,
                                                candidate_binding=candidate_binding)
    baseline_checkpoints = {row.get("checkpoint_sha256") for row in base_maps["maps"]}
    if baseline_checkpoints != {checkpoint_sha256}:
        raise ValueError("baseline manifest checkpoint binding missing or ambiguous")
    identity_sha256 = _identity_sha256(base_stats)
    baseline_image_auroc = _image_auroc(base_stats, normal_label=normal_label, anomaly_label=anomaly_label)
    candidate_image_auroc = _image_auroc(candidate_stats, normal_label=normal_label, anomaly_label=anomaly_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_record = {
        "schema_version": 1,
        "metric_path": metric_path,
        "metric_value": baseline_value,
        "evidence_sha256": _hash(baseline_evidence),
        "manifest_sha256": _hash(baseline_manifest),
        "command": baseline.get("command"),
        "protocol": baseline.get("protocol"),
        "sample_count": len(base_stats),
        "metrics": {
            "localization_au_pro_0_05": {"score": baseline_value, "metric_path": metric_path,
                                          "scope": "pixel_localization"},
            "image_auroc": {"score": baseline_image_auroc, "image_score": "max_raw_map",
                              "implementation": "tie_aware_rank", "scope": "image_level_detection"},
        },
    }
    comparison = {
        "schema_version": 1,
        "metric_path": metric_path,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "absolute_change": candidate_value - baseline_value,
        "relative_ratio": candidate_value / baseline_value if baseline_value else None,
        "change_scope": change_scope,
        "change_description": change_description,
        "baseline_evidence_sha256": _hash(baseline_evidence),
        "candidate_evidence_sha256": _hash(candidate_evidence),
        "baseline_manifest_sha256": _hash(baseline_manifest),
        "candidate_manifest_sha256": _hash(candidate_manifest),
        "classification_status": "NOT_COMPUTED_NO_FROZEN_THRESHOLD",
        "verified_bindings": {"checkpoint_sha256": checkpoint_sha256, "sample_identity_sha256": identity_sha256,
                              "candidate_binding_sha256": _hash(candidate_binding)},
        "metrics": {
            "localization_au_pro_0_05": {"baseline": baseline_value, "candidate": candidate_value,
                                          "absolute_change": candidate_value - baseline_value,
                                          "relative_ratio": candidate_value / baseline_value if baseline_value else None,
                                          "metric_path": metric_path, "scope": "pixel_localization"},
            "image_auroc": {"baseline": baseline_image_auroc, "candidate": candidate_image_auroc,
                              "absolute_change": candidate_image_auroc - baseline_image_auroc,
                              "image_score": "max_raw_map", "implementation": "tie_aware_rank",
                              "scope": "image_level_detection"},
        },
    }

    baseline_path = output_dir / "baseline_evaluation.json"
    comparison_path = output_dir / "evaluation_comparison.json"
    cases_path = output_dir / "error_cases.csv"
    baseline_path.write_text(json.dumps(baseline_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for identity in sorted(base_stats):
        base, current = base_stats[identity], candidate_stats[identity]
        reason = "normal_high_score" if base["label"] == normal_label else "anomaly_low_score"
        rows.append({"image_identity": identity, "label": base["label"], "review_reason": reason,
                     "baseline_mean": base["mean"], "baseline_p99": base["p99"], "baseline_max": base["max"],
                     "candidate_mean": current["mean"], "candidate_p99": current["p99"], "candidate_max": current["max"],
                     "classification_status": "NOT_COMPUTED_NO_FROZEN_THRESHOLD"})
    for label in {row["label"] for row in rows}:
        selected = [row for row in rows if row["label"] == label]
        selected.sort(key=lambda row: row["candidate_p99"], reverse=label == normal_label)
        for rank, row in enumerate(selected, 1):
            row["review_rank"] = rank
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: (row["label"], row["review_rank"])))
    metric_visual, review_visual = _write_visuals(output_dir, comparison, rows, baseline_label, candidate_label)
    return {"baseline": baseline_path, "comparison": comparison_path, "error_cases": cases_path,
            "metric_visual": metric_visual, "review_visual": review_visual}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-evidence", "candidate-evidence", "baseline-manifest", "candidate-manifest", "candidate-binding", "artifact-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metric-path", required=True)
    parser.add_argument("--change-scope", required=True)
    parser.add_argument("--change-description", required=True)
    parser.add_argument("--normal-label", required=True)
    parser.add_argument("--anomaly-label", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_evaluation_history(**vars(args))
    print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
