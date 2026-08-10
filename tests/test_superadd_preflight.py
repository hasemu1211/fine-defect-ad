import hashlib
import json
from dataclasses import asdict

import pytest

from fine_defect_ad.superadd_preflight import (
    ANOMALIB_COMMIT,
    DINO_COMMIT,
    DINO_LICENSE_CONTENT_SHA256,
    DINO_LICENSE_IDENTIFIER,
    DINO_LICENSE_URL,
    EXACT_HPLUS,
    EXACT_HPLUS_MODEL_ID,
    FALLBACK_LABEL,
    PINNED_VITS_MODEL_ID,
    PROBE_PRODUCER_MODULE,
    ChallengerBlocked,
    _probe_producer_source_sha256,
    main,
    run_preflight,
)


def _digest(letter):
    return letter * 64


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value):
    return hashlib.sha256(_canon(value)).hexdigest()


def _setup(tmp_path):
    category = tmp_path / "dataset" / "sheet_metal"
    train = category / "train" / "good"
    train.mkdir(parents=True)
    entries = []
    for index in range(10):
        path = train / f"{index:03d}.png"
        path.write_bytes(f"image-{index}".encode())
        entries.append(
            {
                "path": str(path.relative_to(category)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    identity = {"data": {"train": entries, "validation": []}}
    raw = _canon(identity)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    identity_path = artifacts / f"g002-training-identity-r1-{hashlib.sha256(raw).hexdigest()}.json"
    identity_path.write_bytes(raw)
    for name in ("hplus.bin", "vits.bin"):
        # The helper derives the admission digest from this test artifact.
        (artifacts / name).write_bytes(b"weight")
    return category.parent, artifacts, identity_path, entries


def _fixture(entries):
    return {"entries": entries}


def _weight(artifacts, model_id, filename, available=True):
    model = f"https://huggingface.co/{model_id}"
    if not available:
        return {"model_url": model, "access": "license_required"}

    path = artifacts / filename
    return {
        "model_url": model,
        "download_url": f"{model}/resolve/main/model.safetensors",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "local_path": str(path),
        "access": "available",
    }


def _provenance(artifacts, available=True):
    license_record = {
        "identifier": DINO_LICENSE_IDENTIFIER,
        "url": DINO_LICENSE_URL,
        "acceptance": "accepted",
        "content_sha256": DINO_LICENSE_CONTENT_SHA256,
    }
    if not available:
        license_record = {
            "identifier": DINO_LICENSE_IDENTIFIER,
            "url": DINO_LICENSE_URL,
            "acceptance": "not_accepted",
        }
    return {
        "anomalib_commit": ANOMALIB_COMMIT,
        "dino_commit": DINO_COMMIT,
        "timm_version": "1.0.28",
        "dino_license": license_record,
        "weights": {
            "exact_hplus": _weight(
                artifacts, EXACT_HPLUS_MODEL_ID, "hplus.bin", available
            ),
            "pinned_vits": _weight(
                artifacts, PINNED_VITS_MODEL_ID, "vits.bin", available
            ),
        },
    }


def _safe(entries):
    return tuple(
        sorted(
            (
                {
                    "path_sha256": hashlib.sha256(entry["path"].encode()).hexdigest(),
                    "content_sha256": entry["sha256"],
                }
                for entry in entries
            ),
            key=lambda item: (item["path_sha256"], item["content_sha256"]),
        )
    )


def _probe(entries, weight, status="RESOURCE_FAILURE", **more):
    return {
        "schema_version": 1,
        "status": status,
        "variant": "exact_hplus",
        "recipe_fingerprint": _sha(asdict(EXACT_HPLUS)),
        "fixture_entries_sha256": _sha(_safe(entries)),
        "resolved_weight_sha256": weight,
        "runtime_binding_sha256": _digest("d"),
        "resource": {
            "peak_vram_bytes": 1,
            "peak_host_ram_bytes": 2,
            "seconds_per_image": 3.0,
            "index_growth_bytes": 4,
        },
        "reason": "resource preflight",
        "producer_module": PROBE_PRODUCER_MODULE,
        "producer_source_sha256": _probe_producer_source_sha256(),
        "anomalib_commit": ANOMALIB_COMMIT,
        **more,
    }


def _write_probe(artifacts, payload):
    raw = _canon(payload)
    path = artifacts / f"superadd-hplus-probe-r1-{hashlib.sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _plan(entries, provenance):
    return {"run_id": "offline-r1", "fixture": _fixture(entries), "provenance": provenance}


def _trusted(artifacts, entries, provenance):
    def producer(*_, **__):
        payload = _probe(entries, provenance["weights"]["exact_hplus"]["sha256"])
        reference = _write_probe(artifacts, payload)
        return {**payload, "artifact": reference["path"], "artifact_sha256": reference["sha256"]}

    return producer


class _Proof:
    def __init__(self, root):
        self.roots = {"artifact": str(root)}


class _Lease:
    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


def _writer(path, raw, **_):
    path.write_bytes(raw)
    return {"status": "READY"}


def test_gated_license_stops_without_hash_or_probe(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    result = run_preflight(
        _plan(entries, _provenance(artifacts, False)),
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        anomalib_source=artifacts,
    )
    assert result["status"] == "STOPPED_INCOMPLETE"
    assert result["workflow_status"] == "WEIGHT_ACCESS_REQUIRED"
    assert "content_sha256" not in json.dumps(result["provenance"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda provenance: provenance["weights"]["exact_hplus"].__setitem__(
            "model_url", f"https://huggingface.co/{PINNED_VITS_MODEL_ID}"
        ),
        lambda provenance: provenance["weights"]["pinned_vits"].__setitem__(
            "model_url", f"https://huggingface.co/{EXACT_HPLUS_MODEL_ID}"
        ),
        lambda provenance: provenance["weights"]["exact_hplus"].__setitem__(
            "download_url", "https://example.org/x"
        ),
        lambda provenance: provenance["dino_license"].__setitem__(
            "content_sha256", _digest("f")
        ),
    ],
)
def test_swapped_fake_or_unattested_provenance_blocks(tmp_path, mutate):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    mutate(provenance)
    with pytest.raises(ChallengerBlocked):
        run_preflight(
            _plan(entries, provenance),
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            producer=_trusted(artifacts, entries, provenance),
            anomalib_source=artifacts,
        )


def test_available_weight_requires_actual_local_artifact_bytes(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    (artifacts / "hplus.bin").write_bytes(b"tampered")
    with pytest.raises(ChallengerBlocked):
        run_preflight(
            _plan(entries, provenance),
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            producer=_trusted(artifacts, entries, provenance),
            anomalib_source=artifacts,
        )


def test_signed_official_meta_weight_urls_are_hashed_not_persisted(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    signed_query = "?token=private-signed-token"
    provenance["weights"]["exact_hplus"]["download_url"] = (
        "https://dinov3.llamameta.net/dinov3_vith16plus/"
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
        f"{signed_query}"
    )
    provenance["weights"]["pinned_vits"]["download_url"] = (
        "https://dinov3.llamameta.net/dinov3_vits16/"
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        f"{signed_query}"
    )

    result = run_preflight(
        _plan(entries, provenance),
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        producer=_trusted(artifacts, entries, provenance),
        anomalib_source=artifacts,
    )
    assert result["status"] == "READY"
    assert "private-signed-token" not in json.dumps(result)


@pytest.mark.parametrize(
    "variant,url",
    [
        (
            "exact_hplus",
            "https://untrusted.example/dinov3_vith16plus/"
            "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        ),
        (
            "pinned_vits",
            "https://dinov3.llamameta.net/dinov3_vith16plus/"
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        ),
        (
            "exact_hplus",
            "https://dinov3.llamameta.net/dinov3_vith16plus/wrong.pth",
        ),
    ],
)
def test_meta_weight_url_requires_exact_host_directory_and_filename(
    tmp_path, variant, url
):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    provenance["weights"][variant]["download_url"] = url

    with pytest.raises(ChallengerBlocked):
        run_preflight(
            _plan(entries, provenance),
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            producer=_trusted(artifacts, entries, provenance),
            anomalib_source=artifacts,
        )


def test_only_canonical_producer_probe_unlocks_fallback(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    result = run_preflight(
        _plan(entries, provenance),
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        producer=_trusted(artifacts, entries, provenance),
        anomalib_source=artifacts,
    )
    assert result["status"] == "READY"
    assert result["recipe"]["claim_label"] == FALLBACK_LABEL
    assert result["recipe"]["fp16_status"].startswith("NOT_ADMITTED")


def test_inline_hplus_evidence_and_outside_probe_are_rejected(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    provenance = _provenance(artifacts)
    bad_plan = _plan(entries, provenance)
    bad_plan["hplus_evidence"] = {}
    with pytest.raises(ChallengerBlocked):
        run_preflight(
            bad_plan,
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            producer=_trusted(artifacts, entries, provenance),
            anomalib_source=artifacts,
        )


def test_cli_failure_is_private(tmp_path, capsys):
    missing = tmp_path / "private.json"
    assert (
        main(
            [
                "--plan",
                str(missing),
                "--storage-plan",
                str(missing),
                "--training-identity",
                str(missing),
                "--dataset-root",
                str(tmp_path),
                "--lease-directory",
                str(tmp_path),
                "--anomalib-source",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert str(missing) not in capsys.readouterr().out
