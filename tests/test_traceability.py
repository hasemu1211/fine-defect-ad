from pathlib import Path

import pytest

import fine_defect_ad.traceability as traceability
from fine_defect_ad.traceability import decision_references, validate_traceability


ROOT = Path(__file__).resolve().parents[1]


def test_public_traceability_covers_decisions_and_claims():
    assert validate_traceability(ROOT) == {"decisions": 24, "references": 4, "requirements": 24}


def test_unregistered_decision_reference_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(traceability, "decision_references", lambda _: {"DEC-NOT-REGISTERED"})
    with pytest.raises(ValueError, match="unresolved README/source/config"):
        validate_traceability(ROOT)
