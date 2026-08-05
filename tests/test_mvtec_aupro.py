import hashlib
import os
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fine_defect_ad.mvtec_aupro import EMPTY_MASK_STATUS, RESULT_LABEL, EvaluatorSource, EvaluatorVerificationError, local_au_pro_0_05, verify_evaluator


ARCHIVE = Path(os.environ.get("MVTEC_AD_EVALUATOR_ARCHIVE", ""))


@pytest.fixture
def evaluator_archive():
    if not ARCHIVE.is_file():
        pytest.skip("external official MVTec AD evaluator archive is unavailable")
    return ARCHIVE


def test_archive_hash_and_safe_members_are_required(tmp_path):
    forged = tmp_path / "forged.tar.xz"; forged.write_bytes(b"not the official archive")
    with pytest.raises(EvaluatorVerificationError):
        verify_evaluator(forged)


def test_contract_blocks_comparator_and_reports_empty_masks(tmp_path):
    verified = EvaluatorSource(tmp_path, "root", None, {"pro_curve_util.py": "pinned"})
    with patch("fine_defect_ad.mvtec_aupro.verify_evaluator", return_value=verified):
        result = local_au_pro_0_05([[[0.0, 1.0], [2.0, 3.0]]], [None], tmp_path)
    assert result["status"] == RESULT_LABEL
    assert result["integration_limit"] == 0.05
    assert result["config"] == {
        "pro_integration_limit": 0.05,
        "spatial_alignment": "CALLER_PREALIGNED_REQUIRED",
        "none_masks_restored_to_zeros_only": True,
    }
    assert result["comparator"] is None
    assert result["threshold_metrics"].startswith("BLOCKED")
    assert result["ground_truth_restoration"]["none_masks_restored_to_zeros"] == [0]
    assert result["empty_mask_behavior"] == EMPTY_MASK_STATUS
    assert result["output"] is None


def test_compact_output_omits_large_curve_arrays(tmp_path):
    verified = EvaluatorSource(tmp_path, "root", None, {"pro_curve_util.py": "pinned"})
    with patch("fine_defect_ad.mvtec_aupro.verify_evaluator", return_value=verified), patch(
        "fine_defect_ad.mvtec_aupro._official_functions", return_value=(lambda x, y, x_max: 0.025, lambda *_: (__import__("numpy").array([0., .05]), __import__("numpy").array([0., 1.])))
    ):
        result = local_au_pro_0_05([[[0., 1.], [2., 3.]]], [[[0, 1], [0, 0]]], tmp_path, include_curve=False)
    assert result["output"] == {"au_pro_0_05": .5, "curve_points": 2}


def test_known_arrays_lock_official_au_pro_0_05(evaluator_archive, tmp_path):
    assert verify_evaluator(evaluator_archive).archive_sha256 == hashlib.sha256(evaluator_archive.read_bytes()).hexdigest()
    anomaly_map = [[0.0] * 5 for _ in range(5)]
    anomaly_map[0][0] = 0.9  # highest score is one normal pixel
    anomaly_map[0][1] = 0.8  # followed by the sole positive pixel
    ground_truth = [[0] * 5 for _ in range(5)]
    ground_truth[0][1] = 1
    result = local_au_pro_0_05([anomaly_map], [ground_truth], evaluator_archive)
    output = result["output"]
    assert result["config"]["pro_integration_limit"] == 0.05
    assert output["au_pro_0_05"] == pytest.approx(1 / 6, abs=1e-6)
    assert output["fpr"] == pytest.approx([0.0, 1 / 24, 1 / 24, 1.0, 1.0])
    assert output["pro"] == pytest.approx([0.0, 0.0, 1.0, 1.0, 1.0])
    assert result["source"]["kind"] == "archive"

    with tarfile.open(evaluator_archive, "r:xz") as archive:
        archive.extractall(tmp_path, filter="data")
    root_result = local_au_pro_0_05([anomaly_map], [ground_truth], tmp_path)
    assert root_result["output"]["au_pro_0_05"] == pytest.approx(output["au_pro_0_05"])
    assert root_result["output"]["fpr"] == pytest.approx(output["fpr"])
    assert root_result["output"]["pro"] == pytest.approx(output["pro"])
    assert root_result["source"]["kind"] == "root"
    assert root_result["source"]["archive_sha256"] == hashlib.sha256(evaluator_archive.read_bytes()).hexdigest()
