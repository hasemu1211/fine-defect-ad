"""Fail-closed, offline-only admission for the one SuperADD challenger.

This command never downloads a DINO weight or calculates accuracy. It binds an
exact H+ attempt to verified training-image bytes, then permits one pinned
ViT-S fallback only after an H+ resource/reproduction failure.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .g002_eval_runtime import load_training_identity
from .g002_pilot import _sha256
from .gpu_lock import GpuLease
from .storage import (
    READY,
    Allocation,
    PreflightProof,
    StorageBlocked,
    atomic_write,
    preflight,
)

DECISION_ID = "DEC-MOD-002"
ANOMALIB_COMMIT = "3759687e76395c4d6d239552d3bf6d72e003da78"
DINO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
TIMM_VERSION = "1.0.28"
DINO_LICENSE_URL = (
    f"https://github.com/facebookresearch/dinov3/blob/{DINO_COMMIT}/LICENSE.md"
)
DINO_LICENSE_IDENTIFIER = "DINOv3 License"
DINO_LICENSE_CONTENT_SHA256 = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
EXACT_HPLUS_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
PINNED_VITS_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
OFFICIAL_META_WEIGHT_HOST = "dinov3.llamameta.net"
_OFFICIAL_META_WEIGHT_PATHS = {
    EXACT_HPLUS_MODEL_ID: (
        "/dinov3_vith16plus/"
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    ),
    PINNED_VITS_MODEL_ID: (
        "/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ),
}
EXACT_LABEL = "SuperADD exact H+ recipe"
FALLBACK_LABEL = "SuperADD hardware-constrained variant"
PROBE_PRODUCER_MODULE = "fine_defect_ad.superadd_hplus_probe"
_RESOURCE_FAILURES = frozenset({"RESOURCE_FAILURE", "REPRODUCTION_FAILURE"})
_FAILURE_STATUSES = frozenset({"STOPPED_INCOMPLETE", *_RESOURCE_FAILURES})


def _probe_producer_source_sha256() -> str:
    return _sha256(Path(__file__).with_name("superadd_hplus_probe.py"))


@dataclass(frozen=True)
class Recipe:
    variant: str
    claim_label: str
    backbone: str
    layers: tuple[int, ...]
    input_size: int
    patch_overlap: int
    bank_per_layer: int = 100_000
    subsampling_iterations: int = 100
    precision: str = "fp32"
    anomalib_commit: str = ANOMALIB_COMMIT
    dino_commit: str = DINO_COMMIT


EXACT_HPLUS = Recipe(
    "exact_hplus",
    EXACT_LABEL,
    "vit_huge_plus_patch16_dinov3",
    (7, 15, 23, 31),
    640,
    128,
)
PINNED_VITS = Recipe(
    "pinned_vits",
    FALLBACK_LABEL,
    "vit_small_patch16_dinov3",
    (3, 5, 8, 10),
    448,
    16,
)


class ChallengerBlocked(ValueError):
    """An admission condition is absent or violates the one-challenger rule."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _has_forbidden_signal(value: Any) -> bool:
    forbidden_keys = {
        "test",
        "testpub",
        "testpriv",
        "accuracy",
        "auroc",
        "au_pro",
        "metric",
        "score",
        "label",
    }
    forbidden_values = ("test", "accuracy", "metric", "auroc", "score", "label")

    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden_keys or _has_forbidden_signal(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_forbidden_signal(item) for item in value)
    if isinstance(value, str):
        return any(token in value.lower() for token in forbidden_values)
    return False


def _dataset_category(dataset_root: Path) -> Path:
    root = Path(dataset_root).resolve()
    category = root / "sheet_metal"
    return category if category.is_dir() else root


def validate_fixture(
    fixture: Mapping[str, Any],
    *,
    training_identity: Mapping[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, str], ...]:
    """Verify 10–20 exact `data.train[{path,sha256}]` entries against live bytes."""
    if set(fixture) != {"entries"} or not isinstance(fixture.get("entries"), list):
        raise ChallengerBlocked("fixture must contain only manifest train entries")

    entries = fixture["entries"]
    if not 10 <= len(entries) <= 20:
        raise ChallengerBlocked("fixture requires 10–20 train entries")

    data = training_identity.get("data")
    expected = data.get("train") if isinstance(data, Mapping) else None
    if not isinstance(expected, list):
        raise ChallengerBlocked("training identity has no data.train manifest")

    def pair(row: Any) -> tuple[str, str]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not isinstance(row["sha256"], str)
        ):
            raise ChallengerBlocked("fixture entries must be manifest path/sha256 pairs")
        return row["path"], row["sha256"]

    expected_pairs = {pair(row) for row in expected}
    selected = [pair(row) for row in entries]
    if len(set(selected)) != len(selected) or any(
        entry not in expected_pairs for entry in selected
    ):
        raise ChallengerBlocked("fixture entries must be unique existing training identities")

    category = _dataset_category(dataset_root)
    safe: list[dict[str, str]] = []
    for relative, digest in selected:
        relative_path = Path(relative)
        has_safe_relative_path = (
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and len(relative_path.parts) >= 3
            and relative_path.parts[:2] == ("train", "good")
        )
        if not has_safe_relative_path or not _is_lower_sha256(digest):
            raise ChallengerBlocked("fixture must use lowercase train/good manifest identities")

        source = (category / relative_path).resolve()
        if (
            not source.is_file()
            or not _inside(source, category)
            or _sha256(source) != digest
        ):
            raise ChallengerBlocked("fixture file bytes do not match training identity")
        safe.append(
            {
                "path_sha256": sha256(relative.encode()).hexdigest(),
                "content_sha256": digest,
            }
        )
    return tuple(
        sorted(safe, key=lambda item: (item["path_sha256"], item["content_sha256"]))
    )


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(char in "0123456789abcdef" for char in value)
    )


def _weight_urls(model_id: str) -> tuple[str, str]:
    return (
        f"https://huggingface.co/{model_id}",
        f"https://huggingface.co/{model_id}/resolve/",
    )


def _is_official_weight_download_url(url: str, model_id: str) -> bool:
    _, huggingface_prefix = _weight_urls(model_id)
    if url.startswith(huggingface_prefix):
        return True

    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == OFFICIAL_META_WEIGHT_HOST
        and parsed.path == _OFFICIAL_META_WEIGHT_PATHS[model_id]
        and not parsed.fragment
    )


def validate_provenance(
    provenance: Mapping[str, Any],
) -> tuple[dict[str, str], bool, dict[str, str]]:
    """Bind each recipe to its exact HF model, pinned license revision, and weight state."""
    required = {"anomalib_commit", "dino_commit", "timm_version", "dino_license", "weights"}
    if (
        set(provenance) != required
        or provenance.get("anomalib_commit") != ANOMALIB_COMMIT
        or provenance.get("dino_commit") != DINO_COMMIT
    ):
        raise ChallengerBlocked("pinned anomalib/DINO provenance mismatch")

    license_record = provenance.get("dino_license")
    if (
        not isinstance(license_record, Mapping)
        or license_record.get("identifier") != DINO_LICENSE_IDENTIFIER
        or license_record.get("url") != DINO_LICENSE_URL
    ):
        raise ChallengerBlocked("exact DINOv3 license identity is required")

    accepted = license_record.get("acceptance")
    if accepted == "accepted":
        if (
            set(license_record)
            != {"identifier", "url", "acceptance", "content_sha256"}
            or license_record.get("content_sha256") != DINO_LICENSE_CONTENT_SHA256
        ):
            raise ChallengerBlocked(
                "accepted DINOv3 license requires validated content SHA-256"
            )
    elif accepted == "not_accepted":
        if set(license_record) != {"identifier", "url", "acceptance"}:
            raise ChallengerBlocked("unaccepted DINOv3 license must not claim content bytes")
    else:
        raise ChallengerBlocked("DINOv3 license acceptance state is required")

    if provenance.get("timm_version") != TIMM_VERSION:
        raise ChallengerBlocked("pinned timm and exact DINOv3 license are required")

    weights = provenance["weights"]
    if not isinstance(weights, Mapping) or set(weights) != {"exact_hplus", "pinned_vits"}:
        raise ChallengerBlocked(
            "official weight provenance is required for exact H+ and pinned ViT-S"
        )

    output = {
        "anomalib_commit": ANOMALIB_COMMIT,
        "dino_commit": DINO_COMMIT,
        "timm_version": TIMM_VERSION,
        "dino_license_identifier": DINO_LICENSE_IDENTIFIER,
        "dino_license_url_sha256": _sha(DINO_LICENSE_URL),
        "dino_license_acceptance": accepted,
    }
    if accepted == "accepted":
        output["dino_license_content_sha256"] = license_record["content_sha256"]

    gated = accepted != "accepted"
    resolved: dict[str, str] = {}
    model_ids = {
        "exact_hplus": EXACT_HPLUS_MODEL_ID,
        "pinned_vits": PINNED_VITS_MODEL_ID,
    }
    for variant, model_id in model_ids.items():
        item = weights[variant]
        model_url, _download_prefix = _weight_urls(model_id)
        if not isinstance(item, Mapping) or item.get("model_url") != model_url:
            raise ChallengerBlocked(
                f"{variant} requires its exact official DINO model identity"
            )

        access = item.get("access")
        if access == "license_required":
            if set(item) != {"model_url", "access"}:
                raise ChallengerBlocked(
                    f"{variant} gated provenance must not invent a weight URL/hash"
                )
            gated = True
            output[f"{variant}_model_url_sha256"] = _sha(model_url)
        elif access == "available":
            has_available_fields = set(item) == {
                "model_url",
                "download_url",
                "sha256",
                "local_path",
                "access",
            }
            if (
                accepted != "accepted"
                or not has_available_fields
                or not isinstance(item.get("download_url"), str)
                or not _is_official_weight_download_url(item["download_url"], model_id)
                or not isinstance(item.get("local_path"), str)
            ):
                raise ChallengerBlocked(
                    f"{variant} requires official download URL and accepted license"
                )

            digest = item.get("sha256")
            if not _is_lower_sha256(digest):
                raise ChallengerBlocked(f"{variant} requires lower-case SHA-256")
            resolved[variant] = digest
            output[f"{variant}_weight_sha256"] = digest
            output[f"{variant}_download_url_sha256"] = _sha(item["download_url"])
        else:
            raise ChallengerBlocked(f"{variant} weight access is invalid")
    return output, gated, resolved


def verify_local_weights(
    provenance: Mapping[str, Any], artifact_root: Path, resolved: Mapping[str, str]
) -> None:
    """Availability means a hash-verified local weight under the admitted artifact root."""
    for variant, digest in resolved.items():
        path = Path(provenance["weights"][variant]["local_path"]).resolve()
        if (
            not path.is_file()
            or not _inside(path, artifact_root)
            or _sha256(path) != digest
        ):
            raise ChallengerBlocked(f"{variant} local weight path/hash admission failed")


def _load_hplus_probe(
    reference: Mapping[str, Any],
    *,
    artifact_root: Path,
    fixture_entries: tuple[dict[str, str], ...],
    resolved_weight_sha256: str,
) -> Mapping[str, Any]:
    """Admit only a canonical immutable exact-H+ probe, never caller-authored result data."""
    if (
        set(reference) != {"path", "sha256"}
        or not isinstance(reference.get("path"), str)
        or not _is_lower_sha256(reference.get("sha256"))
    ):
        raise ChallengerBlocked("canonical H+ probe path and SHA-256 are required")

    probe_path = Path(reference["path"]).resolve()
    if (
        not probe_path.is_file()
        or reference["sha256"] not in probe_path.name
        or not _inside(probe_path, artifact_root)
        or _sha256(probe_path) != reference["sha256"]
    ):
        raise ChallengerBlocked("H+ probe artifact path/hash admission failed")

    try:
        raw = probe_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChallengerBlocked("H+ probe artifact is unreadable") from exc

    required = {
        "schema_version",
        "status",
        "variant",
        "recipe_fingerprint",
        "fixture_entries_sha256",
        "resolved_weight_sha256",
        "runtime_binding_sha256",
        "resource",
        "reason",
        "producer_module",
        "producer_source_sha256",
        "anomalib_commit",
    }
    if not isinstance(payload, Mapping) or set(payload) != required or raw != _canonical(payload):
        raise ChallengerBlocked("H+ probe must use canonical immutable schema")

    matches_binding = (
        payload.get("schema_version") == 1
        and payload.get("producer_module") == PROBE_PRODUCER_MODULE
        and payload.get("producer_source_sha256") == _probe_producer_source_sha256()
        and payload.get("anomalib_commit") == ANOMALIB_COMMIT
        and payload.get("variant") == EXACT_HPLUS.variant
        and payload.get("recipe_fingerprint") == _sha(asdict(EXACT_HPLUS))
        and payload.get("fixture_entries_sha256") == _sha(fixture_entries)
        and payload.get("resolved_weight_sha256") == resolved_weight_sha256
        and _is_lower_sha256(payload.get("runtime_binding_sha256"))
    )
    if not matches_binding:
        raise ChallengerBlocked("H+ probe recipe/fixture/weight/runtime binding mismatch")
    if _has_forbidden_signal(payload):
        raise ChallengerBlocked("test/accuracy/metric signals are prohibited")

    _validate_hplus_evidence(payload, fixture_entries)
    return payload


def _validate_hplus_evidence(
    evidence: Mapping[str, Any], fixture_entries: tuple[dict[str, str], ...]
) -> str:
    allowed = {
        "schema_version",
        "status",
        "variant",
        "recipe_fingerprint",
        "fixture_entries_sha256",
        "resolved_weight_sha256",
        "runtime_binding_sha256",
        "resource",
        "reason",
        "producer_module",
        "producer_source_sha256",
        "anomalib_commit",
    }
    matches_recipe = (
        set(evidence) == allowed
        and evidence.get("schema_version") == 1
        and evidence.get("producer_module") == PROBE_PRODUCER_MODULE
        and evidence.get("producer_source_sha256") == _probe_producer_source_sha256()
        and evidence.get("anomalib_commit") == ANOMALIB_COMMIT
        and evidence.get("variant") == EXACT_HPLUS.variant
        and evidence.get("recipe_fingerprint") == _sha(asdict(EXACT_HPLUS))
    )
    if not matches_recipe:
        raise ChallengerBlocked("H+ evidence must identify the exact H+ recipe")

    status = evidence.get("status")
    if status not in {"READY", *_FAILURE_STATUSES}:
        raise ChallengerBlocked("H+ evidence status is invalid")
    if evidence.get("fixture_entries_sha256") != _sha(fixture_entries):
        raise ChallengerBlocked("H+ evidence does not bind the frozen training fixture")

    resource = evidence.get("resource")
    required_resource_fields = {
        "peak_vram_bytes",
        "peak_host_ram_bytes",
        "seconds_per_image",
        "index_growth_bytes",
    }
    if (
        not isinstance(resource, Mapping)
        or set(resource) != required_resource_fields
        or any(
            not isinstance(value, (int, float)) or value < 0
            for value in resource.values()
        )
    ):
        raise ChallengerBlocked("H+ resource evidence is incomplete")
    if not isinstance(evidence.get("reason"), str) or not evidence["reason"]:
        raise ChallengerBlocked("H+ evidence reason is required")
    return status


def _resolve_recipe(
    *,
    fixture: Mapping[str, Any],
    training_identity: Mapping[str, Any],
    dataset_root: Path,
    provenance: Mapping[str, Any],
    verified_probe: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose exact H+ first; permit exactly one pinned fallback on admissible failure."""
    if selection is not None:
        raise ChallengerBlocked("accuracy/test selection is prohibited")

    entries = validate_fixture(
        fixture, training_identity=training_identity, dataset_root=dataset_root
    )
    provenance_record, gated, _resolved_weights = validate_provenance(provenance)
    if gated:
        recipe = EXACT_HPLUS
        state = "WEIGHT_ACCESS_REQUIRED"
        status = "STOPPED_INCOMPLETE"
        limitation = (
            "DINOv3 license acceptance is required before signed weight URL and "
            "verified SHA-256 are available."
        )
    elif verified_probe is None:
        recipe = EXACT_HPLUS
        state = "HPLUS_PROBE_REQUIRED"
        status = "STOPPED_INCOMPLETE"
        limitation = (
            "Canonical exact-H+ resource/reproduction probe artifact is required "
            "before admission."
        )
    else:
        hplus_status = _validate_hplus_evidence(verified_probe, entries)
        if hplus_status == "READY":
            recipe = EXACT_HPLUS
            state = "EXACT_HPLUS_ADMITTED"
            status = "READY"
            limitation = "No fallback is permitted after exact H+ passes."
        elif hplus_status in _RESOURCE_FAILURES:
            recipe = PINNED_VITS
            state = "PINNED_VITS_ADMITTED"
            status = "READY"
            limitation = (
                "Pinned fallback enabled solely by exact H+ resource/reproduction "
                "failure."
            )
        else:
            raise ChallengerBlocked(
                "H+ did not establish a fallback-eligible resource/reproduction failure"
            )

    return {
        "decision_id": DECISION_ID,
        "status": status,
        "workflow_status": state,
        "comparison_status": "INCOMPLETE",
        "offline_only": True,
        "recipe": {
            **asdict(recipe),
            "fp16_status": "NOT_ADMITTED_PENDING_SAME_VARIANT_FP32_FP16_PARITY",
        },
        "fixture": {
            "split": "train",
            "count": len(entries),
            "entries": entries,
            "entries_sha256": _sha(entries),
        },
        "training_identity_sha256": _sha(training_identity),
        "provenance": provenance_record,
        "hplus_probe_sha256": _sha(verified_probe)
        if verified_probe is not None
        else None,
        "limitations": [
            limitation,
            "No weights were downloaded and no test/accuracy comparison was performed.",
        ],
    }


def _admit_storage(plan: Mapping[str, Any]) -> PreflightProof:
    try:
        return preflight(
            run_id=plan["run_id"],
            allocations=[Allocation(**item) for item in plan["allocations"]],
            reserve_bytes=plan["reserve_bytes"],
            reserve_evidence=plan["reserve_evidence"],
        )
    except (KeyError, TypeError) as exc:
        raise ChallengerBlocked("source-backed storage plan is required") from exc


def run_preflight(
    plan: Mapping[str, Any],
    storage_plan: Mapping[str, Any],
    *,
    training_identity_path: Path,
    dataset_root: Path,
    lease_directory: Path,
    admit: Callable[[Mapping[str, Any]], PreflightProof] = _admit_storage,
    writer: Callable[..., Mapping[str, Any]] = atomic_write,
    lease_factory: Callable[..., Any] = GpuLease,
    producer: Callable[..., Mapping[str, Any]] | None = None,
    anomalib_source: Path | None = None,
) -> dict[str, Any]:
    """Verify live train bytes before the lease and persist hash-addressed evidence."""
    if (
        set(plan) != {"run_id", "fixture", "provenance"}
        or not isinstance(plan["run_id"], str)
        or not plan["run_id"]
    ):
        raise ChallengerBlocked("preflight plan fields are invalid")
    if storage_plan.get("run_id") != plan["run_id"]:
        raise ChallengerBlocked("storage plan run_id must match challenger plan")

    proof = admit(storage_plan)
    artifact = Path(proof.roots["artifact"]).resolve()
    lease = Path(lease_directory).resolve()
    if lease == artifact or not _inside(lease, artifact):
        raise ChallengerBlocked("lease_directory must be an artifact-root descendant")

    identity, _run = load_training_identity(training_identity_path, artifact)
    entries = validate_fixture(
        plan["fixture"], training_identity=identity, dataset_root=dataset_root
    )
    provenance_record, gated, resolved_weights = validate_provenance(plan["provenance"])
    if not gated:
        verify_local_weights(plan["provenance"], artifact, resolved_weights)

    if gated:
        verified_probe = None
    else:
        from .superadd_hplus_probe import produce_probe

        selected_producer = producer or produce_probe
        if anomalib_source is None:
            raise ChallengerBlocked("anomalib source is required for exact H+ probe")
        produced = selected_producer(
            plan,
            storage_plan,
            training_identity_path=training_identity_path,
            dataset_root=dataset_root,
            lease_directory=lease,
            anomalib_source=anomalib_source,
            admit=lambda _: proof,
        )
        reference = {
            "path": produced["artifact"],
            "sha256": produced["artifact_sha256"],
        }
        verified_probe = _load_hplus_probe(
            reference,
            artifact_root=artifact,
            fixture_entries=entries,
            resolved_weight_sha256=resolved_weights["exact_hplus"],
        )

    record = _resolve_recipe(
        fixture=plan["fixture"],
        training_identity=identity,
        dataset_root=dataset_root,
        provenance=plan["provenance"],
        verified_probe=verified_probe,
    )
    with lease_factory(lease, plan["run_id"], "superadd-preflight"):
        raw = _canonical({"run_id": plan["run_id"], **record})

    destination = artifact / f"superadd-preflight-{plan['run_id']}-{sha256(raw).hexdigest()}.json"
    outcome = writer(destination, raw, proof=proof, run_id=plan["run_id"], overwrite=False)
    if (
        outcome.get("status") != READY
        or not destination.is_file()
        or destination.read_bytes() != raw
    ):
        raise ChallengerBlocked("immutable preflight evidence write failed")
    return {**json.loads(raw), "artifact_sha256": sha256(raw).hexdigest()}


def _error_fingerprint(exc: Exception) -> str:
    return sha256(f"{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--storage-plan", type=Path, required=True)
    parser.add_argument("--training-identity", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lease-directory", type=Path, required=True)
    parser.add_argument("--anomalib-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_preflight(
            json.loads(args.plan.read_text()),
            json.loads(args.storage_plan.read_text()),
            training_identity_path=args.training_identity,
            dataset_root=args.dataset_root,
            lease_directory=args.lease_directory,
            anomalib_source=args.anomalib_source,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == READY else 2
    except (OSError, json.JSONDecodeError, ChallengerBlocked, StorageBlocked) as exc:
        print(
            json.dumps(
                {
                    "status": "STOPPED_INCOMPLETE",
                    "comparison_status": "INCOMPLETE",
                    "reason": "preflight failed",
                    "exception_type": type(exc).__name__,
                    "exception_fingerprint": _error_fingerprint(exc),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
