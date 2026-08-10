"""Stdlib-only public-release decision traceability checks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .evidence import DER_REQUIRED, validate_decision_register

_DECISION = re.compile(r"\bDEC-(?:[A-Z0-9]+-)+[A-Z0-9]+\b")
_REQUIRED_TRACE = frozenset({"requirement_id", "decision_id", "code", "test", "artifact", "readme_claim"})
_SCAN_ROOTS = (Path("README.md"), Path("src"), Path("configs"), Path("docs"))


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _relative_existing(root: Path, raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "\\")):
        raise ValueError(f"invalid {label} path")
    candidate = (root / raw.split("#", 1)[0]).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository") from exc
    if not candidate.is_file():
        raise ValueError(f"missing {label}: {raw}")
    return candidate


def decision_references(root: Path) -> set[str]:
    """Return every decision token in README, source, and optional config files."""
    files: list[Path] = []
    for item in _SCAN_ROOTS:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file() and child.suffix in {".py", ".toml", ".json", ".yaml", ".yml", ".md", ".txt"})
    references: set[str] = set()
    for path in files:
        try:
            references.update(_DECISION.findall(path.read_text(encoding="utf-8")))
        except UnicodeDecodeError as exc:
            raise ValueError(f"non-text traceability input: {path}") from exc
    return references


def validate_traceability(root: Path | None = None) -> dict[str, int]:
    root = (root or repository_root()).resolve()
    decisions = validate_decision_register(root / "evidence/decision-register.yaml")
    decision_by_id = {str(row["decision_id"]): row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("duplicate decision ID")
    for decision_id, row in decision_by_id.items():
        if set(row) != DER_REQUIRED:
            raise ValueError(f"{decision_id} does not have exactly the required DER fields")
        if not all(isinstance(row[name], str) and row[name].strip() for name in (
            "scope", "decision", "evidence_type", "primary_source", "reference_locator",
            "measurement_artifact", "derivation", "limitations", "reviewed_at", "artifact_or_model_version",
        )):
            raise ValueError(f"{decision_id} has an empty scalar DER field")
        if not all(isinstance(row[name], list) and row[name] for name in ("drivers", "alternatives")):
            raise ValueError(f"{decision_id} has empty DER alternatives/drivers")
        _relative_existing(root, row["measurement_artifact"], "DER measurement artifact")

    refs = decision_references(root)
    unresolved = sorted(refs - decision_by_id.keys())
    if unresolved:
        raise ValueError(f"unresolved README/source/config/docs decision reference: {unresolved}")

    trace = _read_json(root / "evidence/traceability.json", "traceability matrix")
    if not isinstance(trace, dict) or trace.get("schema_version") != 1 or not isinstance(trace.get("requirements"), list):
        raise ValueError("invalid traceability matrix structure")
    mapped: set[str] = set()
    readme = (root / "README.md").read_text(encoding="utf-8")
    for row in trace["requirements"]:
        if not isinstance(row, dict) or set(row) != _REQUIRED_TRACE:
            raise ValueError("traceability row must contain exactly requirement/DEC/code/test/artifact/README claim")
        if not all(isinstance(row[key], str) and row[key] for key in _REQUIRED_TRACE):
            raise ValueError("traceability row contains empty field")
        if row["decision_id"] not in decision_by_id:
            raise ValueError(f"traceability decision missing from DER: {row['decision_id']}")
        if row["requirement_id"] in mapped:
            raise ValueError(f"duplicate requirement: {row['requirement_id']}")
        mapped.add(row["requirement_id"])
        for key in ("code", "test", "artifact"):
            _relative_existing(root, row[key], key)
        if row["readme_claim"] not in readme:
            raise ValueError(f"README claim not found: {row['readme_claim']}")

    missing_material = sorted(set(decision_by_id) - {row["decision_id"] for row in trace["requirements"]})
    if missing_material:
        raise ValueError(f"DER decisions missing traceability mapping: {missing_material}")
    return {"decisions": len(decisions), "references": len(refs), "requirements": len(mapped)}


if __name__ == "__main__":
    print(json.dumps(validate_traceability(), sort_keys=True))
