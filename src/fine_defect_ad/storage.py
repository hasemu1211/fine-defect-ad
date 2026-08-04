"""Fail-closed storage admission for material R0 writes (never repairs storage)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno, hashlib, json, os, tempfile
from pathlib import Path
from typing import Iterable

STORAGE_BLOCKED = 'STORAGE_BLOCKED'; STOPPED_INCOMPLETE = 'STOPPED_INCOMPLETE'; READY = 'READY'; INVALIDATED = 'INVALIDATED'
ROOT_ENV = {'data':'FINE_DEFECT_DATA_ROOT','artifact':'FINE_DEFECT_ARTIFACT_ROOT','cache':'FINE_DEFECT_CACHE_ROOT','package_cache':'FINE_DEFECT_PACKAGE_CACHE_ROOT','temp':'FINE_DEFECT_TEMP_ROOT','venv':'FINE_DEFECT_VENV_ROOT','docker_root':'FINE_DEFECT_DOCKER_ROOT','source':'FINE_DEFECT_SOURCE_ROOT'}
class StorageBlocked(RuntimeError): pass
class RunInvalidated(RuntimeError): pass

@dataclass(frozen=True)
class Allocation:
    root: str; bytes: int | None; kind: str; source: str; component_id: str = ''; exact_uncompressed_bytes_source: str = ''

@dataclass(frozen=True)
class PreflightProof:
    run_id: str; roots: dict[str, str]; fingerprint: str; created_at: str; filesystems: dict
    components: list[dict]; reserve: dict; status: str = READY; ttl_seconds: int = 300

def _mount(path: Path) -> tuple[str, str, str]:
    best = None; target = str(path.resolve())
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1); fields, fs = left.split(), right.split(); point = fields[4].replace('\\040', ' ')
        if target == point or target.startswith(point.rstrip('/') + '/'):
            candidate = (point, fs[0], fields[5]); best = candidate if best is None or len(point) > len(best[0]) else best
    if not best: raise StorageBlocked('mount resolution failed')
    return best

def _under(child: Path, parent: Path) -> bool:
    try: child.resolve().relative_to(parent.resolve()); return True
    except ValueError: return False

def roots_from_env() -> dict[str, Path]:
    missing = [v for v in ['FINE_DEFECT_AUTHORIZED_NTFS_ROOT', *ROOT_ENV.values()] if not os.environ.get(v)]
    if missing: raise StorageBlocked('missing mandatory ' + ', '.join(missing))
    return {key: Path(os.environ[value]).expanduser().resolve() for key, value in ROOT_ENV.items()}

def _fingerprint(roots: dict[str, Path], run_id: str) -> str:
    return hashlib.sha256(json.dumps({'run_id':run_id, 'roots':{k:str(v) for k,v in sorted(roots.items())}}, sort_keys=True).encode()).hexdigest()

def _probe(path: Path) -> None:
    try:
        fd, temp = tempfile.mkstemp(dir=path, prefix='.r0-probe-')
        try:
            os.write(fd, b'R0'); os.fsync(fd); os.close(fd); fd = -1
            final = temp + '.ok'; os.replace(temp, final)
            if Path(final).read_bytes() != b'R0': raise StorageBlocked('write probe readback failed')
            Path(final).unlink()
        finally:
            if fd >= 0: os.close(fd)
            Path(temp).unlink(missing_ok=True)
    except OSError as exc: raise StorageBlocked(f'write probe failed: {exc}') from exc

def _root_evidence(name: str, path: Path, authorized: Path, *, probe: bool) -> dict:
    if not path.is_dir(): raise StorageBlocked(f'{name} root missing')
    mount, fs_type, options = _mount(path)
    if 'rw' not in options.split(','): raise StorageBlocked(f'{name} mount is not rw')
    ntfs = name in {'data', 'artifact'}
    if ntfs and (not _under(path, authorized) or fs_type not in {'ntfs', 'ntfs3', 'fuseblk'}): raise StorageBlocked(f'{name} must be an authorized NTFS descendant')
    if not ntfs and fs_type != 'ext4': raise StorageBlocked(f'{name} must remain on ext4')
    daemon_owned = name == 'docker_root'
    source = name == 'source'
    if source and (not os.access(path, os.R_OK) or not (path / '.git').exists()): raise StorageBlocked('source must be a readable Git checkout')
    if not daemon_owned and not source and not os.access(path, os.W_OK): raise StorageBlocked(f'{name} is not writable')
    if probe and not daemon_owned and not source: _probe(path)
    # A mounted rw filesystem is evidence only of the kernel accepting the mount,
    # not of an NTFS dirty/hibernation-clean state.
    return {'root':str(path), 'mountpoint':mount, 'fs_type':fs_type, 'mount_options':options,
            'mount_rw_accepted':True, 'dirty_state':'UNKNOWN_WHILE_MOUNTED',
            'write_probe':probe and not daemon_owned and not source,
            'access_mode':'daemon-mediated' if daemon_owned else 'readable-git-checkout' if source else 'user-writable'}

def _docker_storage(roots: dict[str, Path]) -> dict:
    """Discover Docker storage; no guessed overlay/containerd paths are accepted."""
    from .runtime import discover_docker_storage
    actual = discover_docker_storage()
    if actual.get('status') != READY: raise StorageBlocked(actual.get('reason', 'docker storage discovery failed'))
    if Path(actual['root']).resolve() != roots['docker_root'].resolve(): raise StorageBlocked('docker root differs from docker info')
    return actual

def _reserve(reserve_bytes: int | None, evidence: dict | None) -> dict:
    keys = {'max_pending_atomic_write_bytes', 'measured_high_water_bytes', 'runtime_or_source_citation'}
    if reserve_bytes is None or reserve_bytes < 0 or not isinstance(evidence, dict) or not keys <= evidence.keys():
        raise StorageBlocked('serialized reserve inputs are required')
    pending, high = evidence['max_pending_atomic_write_bytes'], evidence['measured_high_water_bytes']
    if not isinstance(pending, int) or not isinstance(high, int) or pending < 0 or high < 0 or not isinstance(evidence['runtime_or_source_citation'], str) or not evidence['runtime_or_source_citation']:
        raise StorageBlocked('unknown reserve input')
    if reserve_bytes != pending + high: raise StorageBlocked('reserve bytes do not match serialized inputs')
    return {**evidence, 'reserve_bytes':reserve_bytes}

def _components(allocations: Iterable[Allocation], roots: dict[str, Path]) -> list[dict]:
    seen: dict[str, dict] = {}
    for a in allocations:
        if a.root not in roots or a.kind not in {'persistent','transient'} or a.bytes is None or a.bytes < 0 or not a.component_id or a.source in {'', 'unknown'}:
            raise StorageBlocked('unknown/invalid allocation component')
        if 'gzip isize' in a.source.lower() and not a.exact_uncompressed_bytes_source:
            raise StorageBlocked('gzip ISIZE is not exact uncompressed-byte evidence; provide an exact count source')
        item = {'source':a.source, 'component_id':a.component_id, 'root':a.root, 'kind':a.kind, 'bytes':a.bytes}
        if a.exact_uncompressed_bytes_source: item['exact_uncompressed_bytes_source'] = a.exact_uncompressed_bytes_source
        prior = seen.get(a.component_id)
        if prior is not None and prior != item: raise StorageBlocked(f'conflicting duplicate component {a.component_id}')
        seen[a.component_id] = item
    return [seen[k] for k in sorted(seen)]

def preflight(*, run_id: str, allocations: Iterable[Allocation], reserve_bytes: int | None, reserve_evidence: dict | None = None, roots: dict[str, Path] | None = None) -> PreflightProof:
    if not run_id: raise StorageBlocked('run_id is required')
    active = roots or roots_from_env(); authorized_raw = os.environ.get('FINE_DEFECT_AUTHORIZED_NTFS_ROOT')
    if not authorized_raw: raise StorageBlocked('missing mandatory FINE_DEFECT_AUTHORIZED_NTFS_ROOT')
    authorized = Path(authorized_raw).expanduser().resolve(); reserve = _reserve(reserve_bytes, reserve_evidence)
    fs = {name:_root_evidence(name, path, authorized, probe=True) for name, path in active.items()}
    fs['docker_discovery'] = _docker_storage(active)
    components = _components(allocations, active); pools: dict[str, dict] = {}
    for name, path in active.items():
        st, statfs = os.stat(path), os.statvfs(path)
        pools.setdefault(str(st.st_dev), {'roots':[], 'persistent_bytes':0, 'transient_bytes':0, 'available_bytes':statfs.f_bavail * statfs.f_frsize})['roots'].append(name)
    for item in components:
        pool = pools[str(os.stat(active[item['root']]).st_dev)]
        if item['kind'] == 'persistent': pool['persistent_bytes'] += item['bytes']
        else: pool['transient_bytes'] = max(pool['transient_bytes'], item['bytes'])
    for device, pool in pools.items():
        pool['roots'].sort(); pool['required_bytes'] = pool['persistent_bytes'] + pool['transient_bytes'] + reserve['reserve_bytes']
        pool['reserve'] = reserve
        if pool['available_bytes'] < pool['required_bytes']: raise StorageBlocked(f'capacity insufficient on device {device}')
    return PreflightProof(run_id, {k:str(v) for k,v in active.items()}, _fingerprint(active, run_id), datetime.now(timezone.utc).isoformat(), {**fs, 'devices':pools}, components, reserve)

def require_proof(proof: PreflightProof, *, run_id: str, roots: dict[str, Path] | None = None) -> None:
    active = roots or roots_from_env(); authorized_raw = os.environ.get('FINE_DEFECT_AUTHORIZED_NTFS_ROOT')
    if not authorized_raw: raise StorageBlocked('missing mandatory FINE_DEFECT_AUTHORIZED_NTFS_ROOT')
    if proof.status != READY or proof.run_id != run_id or proof.fingerprint != _fingerprint(active, run_id): raise StorageBlocked('fresh preflight proof does not match this run and roots')
    if (datetime.now(timezone.utc) - datetime.fromisoformat(proof.created_at)).total_seconds() > proof.ttl_seconds: raise StorageBlocked('preflight proof expired')
    if not isinstance(proof.reserve, dict) or not proof.components: raise StorageBlocked('incomplete preflight proof')
    # Repeat every authorization, fs, rw and write-probe check immediately before writes.
    for name, path in active.items(): _root_evidence(name, path, Path(authorized_raw).expanduser().resolve(), probe=True)
    current_docker = _docker_storage(active)
    docker_fields = ('root', 'driver', 'driver_type', 'backing_fs', 'image_store_path')
    if any(proof.filesystems.get('docker_discovery', {}).get(key) != current_docker.get(key) for key in docker_fields):
        raise StorageBlocked('docker storage changed since preflight')
    devices = proof.filesystems.get('devices')
    if not isinstance(devices, dict): raise StorageBlocked('missing device capacity proof')
    for device, pool in devices.items():
        names, required = pool.get('roots'), pool.get('required_bytes')
        if not isinstance(names, list) or not isinstance(required, int) or not names: raise StorageBlocked('invalid device capacity proof')
        for name in names:
            root = active[name]; actual_device = str(os.stat(root).st_dev)
            available = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            if actual_device != device or available < required: raise StorageBlocked('capacity changed since preflight')
    artifact = Path(proof.roots['artifact'])
    if (artifact / f'.invalidated-{run_id}.json').exists(): raise RunInvalidated('run is invalidated; use a new run ID')

def atomic_write(destination: Path, payload: bytes, *, proof: PreflightProof, run_id: str) -> dict:
    require_proof(proof, run_id=run_id); destination = Path(destination).resolve(); artifact = Path(proof.roots['artifact']).resolve()
    if not _under(destination, artifact): raise StorageBlocked('material writes are restricted to artifact root')
    temporary = destination.with_name('.' + destination.name + '.partial')
    try:
        with open(temporary, 'wb') as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination); return {'status':READY, 'run_id':run_id, 'path':str(destination)}
    except OSError as exc:
        if exc.errno != errno.ENOSPC: raise
        marker = artifact / f'.invalidated-{run_id}.json'
        marker.write_text(json.dumps({'run_id':run_id, 'status':INVALIDATED, 'workflow_status':STOPPED_INCOMPLETE, 'cause':'ENOSPC', 'partial_path':str(temporary)}))
        return {'status':INVALIDATED, 'workflow_status':STOPPED_INCOMPLETE, 'run_id':run_id, 'cause':'ENOSPC', 'partial_path':str(temporary)}
