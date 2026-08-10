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


def _write_visuals(output_dir: Path, comparison: dict[str, Any], rows: list[dict[str, Any]],
                   baseline_label: str, candidate_label: str) -> tuple[Path, Path]:
    baseline, candidate = comparison["baseline"], comparison["candidate"]
    maximum = max(baseline, candidate, 1e-12)
    bars = [(baseline_label, baseline, 130, "#64748b"), (candidate_label, candidate, 220, "#2563eb")]
    body = []
    for label, value, y, color in bars:
        width = 480 * value / maximum
        body.append(f'<text x="36" y="{y-12}" class="label">{escape(label)}</text>'
                    f'<rect x="36" y="{y}" width="{width:.2f}" height="34" rx="6" fill="{color}"/>'
                    f'<text x="{min(540, 48+width):.2f}" y="{y+24}" class="value">{value:.6f}</text>')
    metric_svg = output_dir / "metric-comparison.svg"
    metric_svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="320" viewBox="0 0 720 320">'
                          '<style>text{font-family:system-ui,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#526070}.label{font-size:14px;font-weight:600}.value{font-size:13px}</style>'
                          '<rect width="100%" height="100%" fill="#f8fafc"/><text x="36" y="42" class="title">AU-PRO@0.05 평가 비교</text>'
                          f'<text x="36" y="66" class="sub">동일 입력 결속 · 상대 변화 {comparison["relative_ratio"]:.2f}×</text>{"".join(body)}</svg>\n', encoding="utf-8")

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
    baseline_checkpoints = {row.get("checkpoint_sha256") for row in base_maps["maps"]}
    binding = _read(candidate_binding)
    if len(baseline_checkpoints) != 1 or None in baseline_checkpoints:
        raise ValueError("baseline checkpoint binding missing or ambiguous")
    if candidate.get("split_freeze_sha256") != _hash(candidate_binding):
        raise ValueError("candidate evidence does not bind candidate binding file")
    checkpoint_sha256 = next(iter(baseline_checkpoints))
    if binding.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("baseline/candidate checkpoint differs")
    identity_sha256 = _identity_sha256(base_stats)

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
