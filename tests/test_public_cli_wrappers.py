import pytest
from fine_defect_ad import calibrate, evaluate, highres_evaluate

@pytest.mark.parametrize("subject", (evaluate, highres_evaluate, calibrate))
def test_domain_cli_wrapper_dispatches(monkeypatch, subject):
    seen=[]
    monkeypatch.setattr(subject, "_main", lambda argv: seen.append(argv) or 7)
    assert subject.main(["--help"]) == 7
    assert seen == [["--help"]]
