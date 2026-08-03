"""Kernel-backed lease for the entire GPU-heavy benchmark window."""
from __future__ import annotations
from datetime import datetime, timezone
import fcntl, json, os, signal, tempfile
from pathlib import Path

class BusyError(RuntimeError):
    def __init__(self, holder_command: str, holder_run_id: str): self.holder_command, self.holder_run_id = holder_command, holder_run_id
    def __str__(self) -> str: return f'BUSY holder_command={self.holder_command} holder_run_id={self.holder_run_id}'

class GpuLease:
    def __init__(self, directory: Path, run_id: str, command: str):
        self.directory, self.run_id, self.command = Path(directory), run_id, command
        self.lock_path = self.directory / 'gpu-heavy.lock'; self.metadata_path = self.directory / 'gpu-heavy-holder.json'; self._fd = None; self._old_signals = {}
    @staticmethod
    def _now() -> str: return datetime.now(timezone.utc).isoformat()
    def _atomic_json(self, path: Path, data: dict) -> None:
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.gpu-')
        try:
            with os.fdopen(fd, 'w') as stream: json.dump(data, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        finally: Path(temp).unlink(missing_ok=True)
    def _event(self, state: str, **extra: object) -> dict:
        data = {'schema_version':1, 'lock_mode':'fcntl.flock', 'state':state, 'timestamp':self._now(), 'command':self.command, 'run_id':self.run_id, 'pid':os.getpid(), **extra}
        events = self.directory / 'gpu-heavy-events'; events.mkdir(exist_ok=True)
        # Collision-proof enough for an artifact name, and O_EXCL preserves history.
        path = events / f"{data['timestamp'].replace(':','-')}-{os.getpid()}-{state}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream: json.dump(data, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
        return data
    def _holder(self) -> dict:
        try: return json.loads(self.metadata_path.read_text())
        except (OSError, json.JSONDecodeError): return {}
    def _release(self, outcome: str) -> None:
        if self._fd is None: return
        event = self._event('released', outcome=outcome)
        self._atomic_json(self.metadata_path, event)
        fcntl.flock(self._fd, fcntl.LOCK_UN); os.close(self._fd); self._fd = None
    def _signal(self, signum: int, _frame: object) -> None:
        self._release(f'signal:{signum}')
        raise SystemExit(128 + signum)
    def __enter__(self) -> 'GpuLease':
        self.directory.mkdir(parents=True, exist_ok=True); self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try: fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd); self._fd = None; holder = self._holder(); raise BusyError(holder.get('command','unknown'), holder.get('run_id','unknown'))
        event = self._event('acquired', acquired_at=self._now())
        self._atomic_json(self.metadata_path, event)
        for sig in (signal.SIGINT, signal.SIGTERM): self._old_signals[sig] = signal.signal(sig, self._signal)
        return self
    def __exit__(self, exc_type, exc, _traceback) -> None:
        self._release('normal' if exc is None else 'exception')
        for sig, previous in self._old_signals.items(): signal.signal(sig, previous)
        self._old_signals.clear()

def benchmark_window(directory: Path, run_id: str, steps: list[callable]) -> list[object]:
    with GpuLease(directory, run_id, 'benchmark-controller'): return [step() for step in steps]
