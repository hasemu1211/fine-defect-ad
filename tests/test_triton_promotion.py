import json
from pathlib import Path

import pytest

from fine_defect_ad.triton_promotion import (
    INPUT_SHAPE, INSPECTION_UNAVAILABLE, TritonConfig, config_pbtxt,
    direct_client_benchmark, onnx_fallback_evidence, parity_report, unavailable_result,
)


def test_fixed_config_and_pytorch_contract():
    assert 'max_batch_size: 0' in config_pbtxt()
    assert 'pytorch_libtorch' in config_pbtxt()
    assert 'dims: [1, 3, 256, 256]' in config_pbtxt()
    assert 'name: "OUTPUT__0"' in config_pbtxt() and 'name: "OUTPUT__1"' in config_pbtxt()
    from fine_defect_ad.triton_promotion import OUTPUT_NAMES, OUTPUT_SEMANTICS
    assert OUTPUT_NAMES == ('OUTPUT__0', 'OUTPUT__1') and OUTPUT_SEMANTICS == {'OUTPUT__0': 'map_st', 'OUTPUT__1': 'map_stae'}
    with pytest.raises(ValueError): TritonConfig(max_batch_size=1)


def test_onnx_gap_selects_torchscript_only_after_both_gaps():
    evidence = onnx_fallback_evidence(host_onnx_available=False, image_backends=['identity', 'python', 'pytorch'])
    assert evidence['selected_export'] == 'torchscript'
    with pytest.raises(ValueError): onnx_fallback_evidence(host_onnx_available=True, image_backends=['pytorch'])


def test_parity_and_failure_are_fail_closed():
    assert parity_report(([[1]], [[1]]), ([[1]], [[1]]), tolerance={'atol': 0.0, 'rtol': 0.0, 'provenance': 'unit measured envelope'})['status'] == 'PARITY_PASS'
    assert parity_report(([[1]], [[1]]), ([[1]], [[1]]))['cause'] == 'TOLERANCE_EVIDENCE_REQUIRED'
    failed = parity_report(([[1]], [[1]]), ([[2]], [[1]]), tolerance={'atol': 0.0, 'rtol': 0.0, 'provenance': 'unit measured envelope'})
    assert failed['status'] == INSPECTION_UNAVAILABLE and not failed['promotion_eligible']
    assert unavailable_result('timeout')['verdict'] is None


def test_direct_client_benchmark_warms_and_reports_percentiles():
    calls = []
    def caller(*_args, **_kwargs):
        calls.append(1); return {"seconds": len(calls) / 1000}
    result = direct_client_benchmark("http://unused", [0], concurrencies=(1,), warmup_requests=1, samples_per_concurrency=3, caller=caller)
    row = result["rows"][0]
    assert len(calls) == 4 and row["p50_seconds"] <= row["p95_seconds"] <= row["p99_seconds"]
    assert result["queue_time"] == "UNAVAILABLE" and result["not_perf_analyzer"]


def test_torchscript_persistence_never_writes_when_writer_fails(tmp_path):
    from fine_defect_ad.triton_promotion import _persist_torchscript
    class Proof: pass
    destination = tmp_path / 'model.pt'
    with pytest.raises(RuntimeError):
        _persist_torchscript(b'model', destination, proof=Proof(), run_id='unit', writer=lambda *_args, **_kwargs: {'status': 'INVALIDATED'})
    assert not destination.exists()


def test_raw_maps_source_never_inspects_lightning_trainer():
    from fine_defect_ad.triton_promotion import raw_maps_source
    class Core:
        def get_maps(self, _image, *, normalize): return _image, _image
    class LightningWrapper:
        model = Core()
        @property
        def trainer(self):
            raise AssertionError('trainer must not be inspected during export')
    assert isinstance(raw_maps_source(LightningWrapper()), Core)


def test_fixed_adapter_matches_eager_and_traced_cpu():
    torch = pytest.importorskip('torch')
    try:
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
    except ImportError:
        pytest.skip('R1 overlay is only available to the recorded host interpreter')
    from fine_defect_ad.triton_promotion import fixed_256_adapter
    torch.manual_seed(7)
    core = EfficientAdModel(teacher_out_channels=8, model_size=EfficientAdModelSize.S).eval()
    image = torch.rand(1, 3, 256, 256)
    adapter = fixed_256_adapter(core)
    adapter = adapter.to(torch.device('cpu'))
    assert adapter.imagenet_mean.device.type == 'cpu' and adapter.imagenet_std.device.type == 'cpu'
    with torch.inference_mode():
        eager = core.get_maps(image, normalize=False)
        adapted = adapter(image)
        traced = torch.jit.trace(adapter, image, check_trace=False)(image)
    for expected, actual, exported in zip(eager, adapted, traced):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
        assert torch.allclose(actual, exported, atol=1e-6, rtol=1e-5)


def test_cpu_export_does_not_mutate_caller_model_device(tmp_path):
    torch = pytest.importorskip('torch')
    try:
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
    except ImportError:
        pytest.skip('R1 overlay is only available to the recorded host interpreter')
    from fine_defect_ad.triton_promotion import save_torchscript
    core = EfficientAdModel(teacher_out_channels=8, model_size=EfficientAdModelSize.S).eval()
    original_device = next(core.parameters()).device
    class Proof: pass
    def writer(destination, payload, **_kwargs):
        Path(destination).write_bytes(payload); return {'status': 'READY'}
    artifact = save_torchscript(core, tmp_path / 'model.pt', torch.rand(1, 3, 256, 256), proof=Proof(), run_id='unit', writer=writer)
    assert next(core.parameters()).device == original_device
    loaded = torch.jit.load(artifact['path'])
    with torch.inference_mode(): assert len(loaded(torch.rand(1, 3, 256, 256))) == 2


def test_fixed_adapter_rejects_non_small_candidate():
    torch = pytest.importorskip('torch')
    try:
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
    except ImportError:
        pytest.skip('R1 overlay is only available to the recorded host interpreter')
    from fine_defect_ad.triton_promotion import fixed_256_adapter
    with pytest.raises(ValueError, match='EfficientAD-S'):
        fixed_256_adapter(EfficientAdModel(teacher_out_channels=8, model_size=EfficientAdModelSize.M).eval())


def test_public_promotion_preflights_before_any_effect_and_persists_failure(tmp_path):
    from fine_defect_ad.triton_promotion import PromotionArgs, run_promotion
    dataset = tmp_path / 'dataset' / 'sheet_metal' / 'train' / 'good'; dataset.mkdir(parents=True)
    source = dataset / 'sample.png'; source.write_bytes(b'x')
    checkpoint = tmp_path / 'checkpoint.ckpt'; checkpoint.write_bytes(b'checkpoint')
    events = []
    class Proof: roots = {'artifact': str(tmp_path)}
    def admit(**_): events.append('preflight'); return Proof()
    def runner(*_args, **_kwargs):
        assert events == ['preflight']; events.append('probe')
        return type('Result', (), {'returncode': 0, 'stdout': 'identity\npython\npytorch\n', 'stderr': ''})()
    def writer(destination, payload, **_):
        Path(destination).write_bytes(payload); return {'status': 'READY'}
    result = run_promotion(PromotionArgs(artifact_root=tmp_path, checkpoint=checkpoint, metrics=checkpoint, final_attempt=checkpoint, training_identity=checkpoint, dataset_root=tmp_path / 'dataset', teacher_small=checkpoint, imagenette_root=tmp_path, lease_directory=tmp_path, source_image=source, split_freeze=checkpoint, run_id='unit'), admit=admit, writer=writer, runner=runner, steps=lambda *_: {'status': 'injected', 'model': {'path': '/private/model.pt', 'sha256': 'a' * 64, 'bytes': 3}})
    assert events == ['preflight', 'probe'] and result['status'] == INSPECTION_UNAVAILABLE
    assert Path(result['artifact']).is_file()
    binding = result['binding']; assert binding['source_image'] == 'sheet_metal/train/good/sample.png' and len(binding['checkpoint_sha256']) == 64
    assert result['stage'] == 'hardware_steps' and binding['mode']['image'].endswith('d624db3a')
    assert result['steps']['model'] == {'sha256': 'a' * 64, 'bytes': 3}


def test_public_parser_and_source_privacy_reject_test_partition(tmp_path):
    from fine_defect_ad.triton_promotion import PromotionArgs, _admit_source_image, g002_runtime_args, parse_promotion_args
    parsed = parse_promotion_args(['--artifact-root', str(tmp_path), '--dataset-root', str(tmp_path), '--source-image', str(tmp_path / 'x'), '--checkpoint', str(tmp_path / 'c'), '--metrics', str(tmp_path / 'm'), '--final-attempt', str(tmp_path / 'f'), '--training-identity', str(tmp_path / 'i'), '--split-freeze', str(tmp_path / 's'), '--teacher-small', str(tmp_path / 't'), '--imagenette-root', str(tmp_path), '--lease-directory', str(tmp_path), '--run-id', 'safe'])
    assert parsed.run_id == 'safe'
    assert g002_runtime_args(parsed).run_id == 'safe'
    image = tmp_path / 'sheet_metal' / 'test_public' / 'good' / 'x.png'; image.parent.mkdir(parents=True); image.write_bytes(b'x')
    with pytest.raises(ValueError): _admit_source_image(PromotionArgs(artifact_root=tmp_path, checkpoint=tmp_path / 'c', metrics=tmp_path / 'm', final_attempt=tmp_path / 'f', training_identity=tmp_path / 'i', dataset_root=tmp_path, teacher_small=tmp_path / 't', imagenette_root=tmp_path, lease_directory=tmp_path, source_image=image, split_freeze=tmp_path / 's', run_id='safe'))


def test_probe_requires_verified_pytorch_backend():
    from fine_defect_ad.triton_promotion import runtime_probes
    result = type('Result', (), {'returncode': 0, 'stdout': 'python\n', 'stderr': ''})()
    with pytest.raises(RuntimeError): runtime_probes(runner=lambda *_args, **_kwargs: result)


def test_promotion_rejects_lease_outside_admitted_artifact_root(tmp_path):
    from fine_defect_ad.triton_promotion import PromotionArgs, run_promotion
    train = tmp_path / 'dataset' / 'sheet_metal' / 'train' / 'good'; train.mkdir(parents=True)
    source, checkpoint = train / 'x.png', tmp_path / 'checkpoint'; source.write_bytes(b'x'); checkpoint.write_bytes(b'c')
    args = PromotionArgs(tmp_path, checkpoint, checkpoint, checkpoint, checkpoint, tmp_path / 'dataset', checkpoint, tmp_path, tmp_path.parent, source, checkpoint, 'unit')
    class Proof: roots = {'artifact': str(tmp_path)}
    with pytest.raises(ValueError, match='lease directory'):
        run_promotion(args, admit=lambda **_: Proof(), runner=lambda **_: None)
