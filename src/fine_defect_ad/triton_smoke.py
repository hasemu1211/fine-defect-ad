"""One-container Triton GPU smoke test for the pinned R0 image."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from .evidence import validate_evidence
from .gpu_lock import BusyError, GpuLease
from .storage import Allocation, preflight, require_proof

IMAGE = 'nvcr.io/nvidia/tritonserver@sha256:80caf7d0be25520d39c5162cdeec1f6b2febe4ab774d7b25138cd602d624db3a'

_CONTAINER_PROGRAM = r"""
import json, os, subprocess, time, urllib.request
from pathlib import Path
def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader'], text=True).strip()
baseline, peak, response, failure = gpu(), 0, None, None
server = subprocess.Popen(['tritonserver', '--model-repository=/models', '--log-verbose=0'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(60):
        peak = max(peak, int(gpu().split(',')[-1].split()[0]))
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v2/health/ready', timeout=1) as r:
                if r.status == 200: break
        except OSError: time.sleep(.5)
    else: raise RuntimeError('Triton readiness timeout')
    body = json.dumps({'inputs':[{'name':'INPUT__0','shape':[1,4],'datatype':'FP32','data':[1.0,2.0,3.0,4.0]}], 'outputs':[{'name':'OUTPUT__0'}]}).encode()
    request = urllib.request.Request('http://127.0.0.1:8000/v2/models/add_one/infer', data=body, headers={'Content-Type':'application/json'})
    response = json.loads(urllib.request.urlopen(request, timeout=10).read())
    values = response['outputs'][0]['data']; assert values == [2.0, 3.0, 4.0, 5.0], values
    peak = max(peak, int(gpu().split(',')[-1].split()[0]))
except Exception as exc:
    failure = repr(exc) + (': ' + exc.read().decode() if hasattr(exc, 'read') else '')
finally:
    server.terminate()
    try: logs, _ = server.communicate(timeout=10)
    except subprocess.TimeoutExpired: server.kill(); logs, _ = server.communicate()
print(json.dumps({'response': response, 'failure': failure, 'gpu_baseline': baseline, 'gpu_peak_used_mib': peak, 'triton_version': os.environ.get('TRITON_SERVER_VERSION'), 'cuda_version': os.environ.get('CUDA_VERSION'), 'backend_inventory': [line for line in logs.splitlines() if 'backend' in line.lower()], 'server_log': logs}))
if failure: raise SystemExit(1)
"""

_MODEL_PROGRAM = """import sys
from pathlib import Path
import torch
repo = Path(sys.argv[1]) / 'add_one' / '1'; repo.mkdir(parents=True, exist_ok=True)
class AddOne(torch.nn.Module):
    def forward(self, x): return x + 1
torch.jit.trace(AddOne().eval(), torch.tensor([[1., 2., 3., 4.]])).save(str(repo / 'model.pt'))
(repo.parent / 'config.pbtxt').write_text('name: "add_one"\\nplatform: "pytorch_libtorch"\\nmax_batch_size: 1\\ninput [{ name: "INPUT__0" data_type: TYPE_FP32 dims: [4] }]\\noutput [{ name: "OUTPUT__0" data_type: TYPE_FP32 dims: [4] }]\\ninstance_group [{ kind: KIND_GPU count: 1 }]\\n')
print(torch.__version__, torch.version.cuda)
"""

def build_container_command(model_repo: Path) -> list[str]:
    return ['docker', 'run', '--rm', '--gpus', 'all', '--log-driver=none', '--read-only',
            '--tmpfs', '/tmp:rw,size=64m,mode=1777', '--tmpfs', '/run:rw,size=16m',
            '--tmpfs', '/root/.cache:rw,size=64m', '-v', f'{model_repo}:/models:rw',
            '--entrypoint', 'python3', IMAGE, '-c', _CONTAINER_PROGRAM]

def _image() -> dict:
    result = subprocess.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{json .}}'], text=True, capture_output=True, check=True)
    info = json.loads(result.stdout)
    return {'id': info['Id'], 'repo_digests': info.get('RepoDigests', []), 'size_bytes': info['Size']}

def _write_model(model_repo: Path) -> str:
    torch_python = os.environ.get('FINE_DEFECT_HOST_PYTHON')
    if not torch_python: raise RuntimeError('FINE_DEFECT_HOST_PYTHON is required to create the TorchScript smoke model')
    result = subprocess.run([torch_python, '-c', _MODEL_PROGRAM, str(model_repo)], text=True, capture_output=True, check=True)
    return result.stdout.strip()

def run_smoke(artifact_root: Path, run_id: str, plan_path: Path) -> dict:
    artifact_root = Path(artifact_root); model_repo = artifact_root / 'triton-r0-smoke-model-repo'; raw_dir = artifact_root / 'triton-r0-smoke-raw'
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('run_id') != run_id: raise RuntimeError('smoke storage plan run_id does not match')
    proof = preflight(run_id=run_id, allocations=[Allocation(**item) for item in plan['allocations']], reserve_bytes=plan['reserve_bytes'], reserve_evidence=plan['reserve_evidence'])
    require_proof(proof, run_id=run_id)
    model_repo.mkdir(parents=True, exist_ok=True); raw_dir.mkdir(parents=True, exist_ok=True)
    host_torch = _write_model(model_repo)
    command = build_container_command(model_repo)
    with GpuLease(artifact_root, run_id, 'triton-r0-smoke'):
        try:
            with GpuLease(artifact_root, run_id + '-contender', 'triton-r0-smoke-contention-probe'):
                raise AssertionError('contention probe acquired the primary lease')
        except BusyError as exc:
            contention = {'status': 'BUSY', 'message': str(exc)}
        require_proof(proof, run_id=run_id)
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=90)
    lease_events = []
    for path in (artifact_root / 'gpu-heavy-events').glob('*.json'):
        event = json.loads(path.read_text())
        if event.get('run_id') == run_id: lease_events.append({key: event[key] for key in ('state', 'timestamp', 'outcome') if key in event})
    raw = {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (raw_dir / f'{run_id}-{attempt}.json').write_text(json.dumps(raw, sort_keys=True))
    if result.returncode: raise RuntimeError(f'Triton smoke failed: {result.stderr or result.stdout}')
    container = json.loads(result.stdout)
    record = {'run_id': run_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
              'command': 'docker run --rm --gpus all --log-driver=none <pinned-triton-digest> python3 -c <R0 smoke>',
              'status': 'READY', 'limitations': ['single fixed-shape add-one smoke; not a performance or production serving claim'],
              'lock_mode': 'fcntl.flock', 'image': _image(), 'container': {**container, 'model_generator_torch_version': host_torch}, 'contention': contention,
              'lease_events': sorted(lease_events, key=lambda event: event['timestamp']),
              'storage_preflight': {'run_id': proof.run_id, 'fingerprint': proof.fingerprint, 'created_at': proof.created_at}}
    validate_evidence(record)
    return record

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('--artifact-root', required=True); parser.add_argument('--plan', required=True); parser.add_argument('--run-id', default='triton-r0-smoke')
    args = parser.parse_args(argv)
    record = run_smoke(Path(args.artifact_root), args.run_id, Path(args.plan))
    print(json.dumps(record, sort_keys=True)); return 0

if __name__ == '__main__': raise SystemExit(main())
