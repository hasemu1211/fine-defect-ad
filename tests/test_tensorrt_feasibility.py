import json
from pathlib import Path
import pytest
from fine_defect_ad.tensorrt_feasibility import TensorRTArgs, fp32_parity, trtexec_command, trtexec_compare_command


def args(tmp_path):
    return TensorRTArgs(tmp_path,tmp_path/'model.onnx',tmp_path/'model.plan',tmp_path/'in.bin',tmp_path/'out.bin',tmp_path/'manifest.json',tmp_path,.5,'unit')

def test_fp32_trtexec_dynamic_batch_commands(tmp_path):
    build=trtexec_command(args(tmp_path)); compare=trtexec_compare_command(args(tmp_path))
    assert '--minShapes=INPUT__0:1x3x256x256' in build and '--optShapes=INPUT__0:4x3x256x256' in build and '--maxShapes=INPUT__0:8x3x256x256' in build
    assert '--fp16' not in build and any(item.startswith('--loadInputs=INPUT__0:') for item in compare)

def test_final_contract_reuses_combined_parity():
    np=pytest.importorskip('numpy'); entries=[{'path':str(i),'sha256':'x'} for i in range(3)]
    result=fp32_parity(np.zeros((1,)),np.zeros((1,)),np.zeros((1,)),np.zeros((1,)),entries=entries,eager_final=lambda _:np.array([[.1]]),trt_final=lambda _:np.array([[.1]]),threshold=.5)
    assert result['raw']['st']['status']=='NUMERICAL_DIAGNOSTIC' and result['final']['status']=='PARITY_PASS'

def test_onnx_export_rejects_noncanonical_tensor_without_torch():
    from fine_defect_ad.tensorrt_feasibility import export_from_admitted_model
    class Bad: ndim=3; shape=(3,256,256)
    with pytest.raises(ValueError): export_from_admitted_model(None,Bad(),Path('/tmp/no.onnx'),proof=object(),run_id='x')


def test_parser_reuses_g002_arguments_and_accepts_source(tmp_path):
    from fine_defect_ad.tensorrt_feasibility import parse_args
    value = parse_args(['--artifact-root', str(tmp_path), '--checkpoint', str(tmp_path/'c'), '--metrics', str(tmp_path/'m'), '--final-attempt', str(tmp_path/'f'), '--training-identity', str(tmp_path/'i'), '--dataset-root', str(tmp_path), '--teacher-small', str(tmp_path/'t'), '--imagenette-root', str(tmp_path), '--lease-directory', str(tmp_path), '--run-id', 'unit', '--source-image', str(tmp_path/'source.png'), '--split-freeze', str(tmp_path/'freeze.json'), '--parity-manifest', str(tmp_path/'parity.json')])
    assert value.run_id == 'unit' and value.source_image.name == 'source.png'


def test_parse_trtexec_output_named_branches():
    from fine_defect_ad.tensorrt_feasibility import parse_trtexec_output
    output = parse_trtexec_output(json.dumps({'outputs': [{'name': 'OUTPUT__0', 'shape': [1, 1, 1, 2], 'data': [1, 2]}, {'name': 'OUTPUT__1', 'shape': [1, 1, 1, 2], 'data': [3, 4]}]}))
    assert output['OUTPUT__0'].shape == (1, 1, 1, 2)
    assert output['OUTPUT__1'].tolist() == [[[[3.0, 4.0]]]]


def test_trtexec_commands_mount_only_artifact_children(tmp_path):
    from fine_defect_ad.tensorrt_feasibility import TRTEXEC
    command = trtexec_command(args(tmp_path))
    assert command[:8] == ['docker', 'run', '--rm', '--gpus', 'all', '-v', f'{tmp_path.resolve()}:/work', 'nvcr.io/nvidia/tritonserver@sha256:80caf7d0be25520d39c5162cdeec1f6b2febe4ab774d7b25138cd602d624db3a']
    assert command[8] == TRTEXEC
    assert '--onnx=/work/model.onnx' in command and all(str(tmp_path) not in item for item in command[8:])


def test_trtexec_commands_reject_nested_artifact_paths(tmp_path):
    bad = TensorRTArgs(tmp_path, tmp_path/'nested'/'model.onnx', tmp_path/'model.plan', tmp_path/'in.bin', tmp_path/'out.bin', tmp_path/'manifest.json', tmp_path, .5, 'unit')
    with pytest.raises(ValueError, match='direct children'):
        trtexec_command(bad)


def test_parse_trtexec_output_accepts_real_tensorrt11_dimensions_schema():
    from fine_defect_ad.tensorrt_feasibility import parse_trtexec_output
    output = parse_trtexec_output(json.dumps([{'name': 'OUTPUT__0', 'dimensions': '1x1x1x2', 'values': [1, 2]}, {'name': 'OUTPUT__1', 'dimensions': '1x1x1x2', 'values': [3, 4]}]))
    assert output['OUTPUT__0'].shape == (1, 1, 1, 2)
