"""Command-derived runtime evidence; unavailable checks stay pending."""
from __future__ import annotations
import json, os, subprocess

READY='READY'; STOPPED_INCOMPLETE='STOPPED_INCOMPLETE'

def _run(command: list[str]) -> dict:
    if any(os.path.basename(part) == 'sudo' for part in command): raise ValueError('sudo is forbidden')
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        return {'command':command, 'returncode':result.returncode, 'stdout':result.stdout.strip(), 'stderr':result.stderr.strip()}
    except OSError as exc: return {'command':command, 'returncode':127, 'stdout':'', 'stderr':str(exc)}

def discover_docker_storage() -> dict:
    evidence = _run(['docker', 'info', '--format', '{{json .}}'])
    if evidence['returncode'] != 0: return {'status':STOPPED_INCOMPLETE, 'reason':'DOCKER_INFO_UNAVAILABLE', 'evidence':evidence}
    try: info = json.loads(evidence['stdout'])
    except json.JSONDecodeError: return {'status':STOPPED_INCOMPLETE, 'reason':'DOCKER_INFO_NOT_JSON', 'evidence':evidence}
    root, driver = info.get('DockerRootDir'), info.get('Driver')
    status = dict(info.get('DriverStatus') or [])
    # Docker reports a backing filesystem but does not promise that an overlay2
    # directory is its active containerd image-store path.
    if not isinstance(root, str) or not root or not isinstance(driver, str) or not driver:
        return {'status':STOPPED_INCOMPLETE, 'reason':'DOCKER_STORAGE_FIELDS_MISSING', 'evidence':evidence}
    access_mode = 'rootless' if 'name=rootless' in info.get('SecurityOptions', []) else 'no_sudo_non_rootless'
    return {'status':READY, 'root':root, 'driver':driver, 'driver_type':status.get('driver-type'), 'access_mode':access_mode,
            'backing_fs':status.get('Backing Filesystem'), 'image_store_path':'UNKNOWN_CONTAINED_BY_DOCKER_ROOT',
            'limitations':['Docker 29 containerd image-store child path is not inferred; DockerRootDir is the storage boundary.'], 'evidence':evidence}

def _host_tensor_smoke() -> dict:
    python = os.environ.get('FINE_DEFECT_HOST_PYTHON')
    if not python: return {'status':'PENDING', 'reason':'FINE_DEFECT_HOST_PYTHON_UNSET'}
    result = _run([python, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.mem_get_info()[1])'])
    return {'status':READY if result['returncode'] == 0 else 'PENDING', 'evidence':result}

def container_gpu_spike(image: str | None) -> dict:
    if not image: return {'status':STOPPED_INCOMPLETE, 'reason':'NO_STORAGE_APPROVED_CUDA_IMAGE'}
    result = _run(['docker', 'run', '--rm', '--gpus', 'all', image, 'python', '-c', 'import torch; assert torch.cuda.is_available(); print(torch.empty((1024,1024),device="cuda").sum())'])
    if result['returncode'] != 0 and ('CDI' in result['stderr'] or 'cdi' in result['stderr']):
        return {'status':STOPPED_INCOMPLETE, 'reason':'DOCKER_CDI_GPU_UNAVAILABLE', 'evidence':result}
    return {'status':READY if result['returncode'] == 0 else STOPPED_INCOMPLETE,
            'reason':None if result['returncode'] == 0 else 'CONTAINER_GPU_SPIKE_FAILED', 'evidence':result}

def collect_runtime_evidence(*, approved_cuda_image: str | None = None) -> dict:
    gpu = _run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    docker = discover_docker_storage(); host = _host_tensor_smoke()
    container = container_gpu_spike(approved_cuda_image)
    ready = gpu['returncode'] == 0 and docker['status'] == READY and host['status'] == READY and container['status'] == READY
    return {'status':READY if ready else STOPPED_INCOMPLETE,
            'reason':None if ready else container.get('reason') or docker.get('reason') or host.get('reason') or 'HOST_GPU_UNAVAILABLE',
            'host':{'gpu_inventory':gpu, 'tensor_smoke':host}, 'docker_storage':docker, 'container':container}
