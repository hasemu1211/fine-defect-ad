import sys
import subprocess
import types
from pathlib import Path

import pytest

from fine_defect_ad import superadd_hplus_probe as subject
from fine_defect_ad.superadd_preflight import (
    PROBE_PRODUCER_MODULE,
    ChallengerBlocked,
    run_preflight,
)
from test_superadd_preflight import _Lease, _Proof, _plan, _provenance, _setup, _writer


def test_producer_injected_real_step_writes_consumer_schema(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    plan = _plan(entries, _provenance(artifacts))
    source = tmp_path / "source"
    source.mkdir()
    seen = []

    def step(**kwargs):
        seen.extend(item["path"] for item in kwargs["fixture"]["entries"])
        return {
            "resource": {
                "peak_vram_bytes": 1,
                "peak_host_ram_bytes": 2,
                "seconds_per_image": 0.1,
                "index_growth_bytes": 3,
            }
        }

    result = subject.produce_probe(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        anomalib_source=source,
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        step=step,
        runtime_binding=lambda _: "d" * 64,
        source_verifier=lambda _: None,
    )
    assert seen == [item["path"] for item in entries]
    assert result["producer_module"] == PROBE_PRODUCER_MODULE
    assert Path(result["artifact"]).is_file()


def test_default_cli_dispatches_producer(tmp_path, monkeypatch, capsys):
    for name in ("p", "s", "i", "d", "l", "a"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        subject,
        "produce_probe",
        lambda *args, **kwargs: {"status": "READY", "artifact": "/private"},
    )
    assert (
        subject.main(
            [
                "--plan",
                str(tmp_path / "p"),
                "--storage-plan",
                str(tmp_path / "s"),
                "--training-identity",
                str(tmp_path / "i"),
                "--dataset-root",
                str(tmp_path / "d"),
                "--lease-directory",
                str(tmp_path / "l"),
                "--anomalib-source",
                str(tmp_path / "a"),
            ]
        )
        == 0
    )
    assert "private" not in capsys.readouterr().out


def test_actual_cuda_oom_is_resource_failure_and_integrates_to_fallback(
    tmp_path, monkeypatch
):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    plan = _plan(entries, _provenance(artifacts))
    source = tmp_path / "source"
    source.mkdir()

    class OOM(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=OOM)),
    )

    def oom_step(**_):
        raise OOM("allocator exhausted")

    result = subject.produce_probe(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        anomalib_source=source,
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        step=oom_step,
        runtime_binding=lambda _: "d" * 64,
        source_verifier=lambda _: None,
    )
    assert result["status"] == "RESOURCE_FAILURE"

    def trusted(*_, **__):
        return result

    selected = run_preflight(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        producer=trusted,
        anomalib_source=source,
    )
    assert selected["workflow_status"] == "PINNED_VITS_ADMITTED"


def test_reproduction_failure_from_pinned_source_integrates_to_fallback(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    plan = _plan(entries, _provenance(artifacts))
    source = tmp_path / "source"
    source.mkdir()

    def source_mismatch(_):
        raise subject.ReproductionFailure("ANOMALIB_SOURCE_REVISION_MISMATCH")

    result = subject.produce_probe(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        anomalib_source=source,
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        runtime_binding=lambda _: "d" * 64,
        source_verifier=source_mismatch,
    )
    assert result["status"] == "REPRODUCTION_FAILURE"

    selected = run_preflight(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        producer=lambda *_, **__: result,
        anomalib_source=source,
    )
    assert selected["workflow_status"] == "PINNED_VITS_ADMITTED"


def test_unknown_probe_failure_remains_stopped_incomplete(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    plan = _plan(entries, _provenance(artifacts))
    source = tmp_path / "source"
    source.mkdir()

    result = subject.produce_probe(
        plan,
        {"run_id": "offline-r1"},
        training_identity_path=identity,
        dataset_root=dataset,
        lease_directory=artifacts / "lease",
        anomalib_source=source,
        admit=lambda _: _Proof(artifacts),
        writer=_writer,
        lease_factory=_Lease,
        step=lambda **_: (_ for _ in ()).throw(RuntimeError("unexpected")),
        runtime_binding=lambda _: "d" * 64,
        source_verifier=lambda _: None,
    )
    assert result["status"] == "STOPPED_INCOMPLETE"


def test_runtime_binding_failure_blocks_preflight_without_fallback(tmp_path):
    dataset, artifacts, identity, entries = _setup(tmp_path)
    plan = _plan(entries, _provenance(artifacts))
    private_source = tmp_path / "private-anomalib-source"

    def unavailable_binding(_):
        raise subprocess.CalledProcessError(1, ["git", "-C", str(private_source)])

    def unavailable_producer(*_, **__):
        return subject.produce_probe(
            plan,
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            anomalib_source=private_source,
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            runtime_binding=unavailable_binding,
            source_verifier=lambda _: None,
        )

    with pytest.raises(ChallengerBlocked, match="RUNTIME_BINDING_UNAVAILABLE"):
        run_preflight(
            plan,
            {"run_id": "offline-r1"},
            training_identity_path=identity,
            dataset_root=dataset,
            lease_directory=artifacts / "lease",
            admit=lambda _: _Proof(artifacts),
            writer=_writer,
            lease_factory=_Lease,
            producer=unavailable_producer,
            anomalib_source=private_source,
        )


def test_cli_unexpected_failure_is_private(tmp_path, monkeypatch, capsys):
    for name in ("p", "s", "i", "d", "l", "a"):
        (tmp_path / name).write_text("{}")
    private_source = tmp_path / "private-anomalib-source"

    def unavailable(*_, **__):
        raise subprocess.CalledProcessError(1, ["git", "-C", str(private_source)])

    monkeypatch.setattr(subject, "produce_probe", unavailable)
    assert (
        subject.main(
            [
                "--plan",
                str(tmp_path / "p"),
                "--storage-plan",
                str(tmp_path / "s"),
                "--training-identity",
                str(tmp_path / "i"),
                "--dataset-root",
                str(tmp_path / "d"),
                "--lease-directory",
                str(tmp_path / "l"),
                "--anomalib-source",
                str(tmp_path / "a"),
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert str(private_source) not in output
    assert "STOPPED_INCOMPLETE" in output
