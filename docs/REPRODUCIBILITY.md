# 재현성

재현 가능한 범위는 공개 코드·고정된 provenance·별도 권한으로 획득한 입력이 있는 환경에서의 증거 검증입니다. 데이터, 모델 가중치, TensorRT plan, GPU/driver는 저장소에 포함되지 않습니다.

1. Python 환경에서 `python3 -m pytest -q`를 실행합니다.
2. 필요 입력을 별도 권한으로 준비한 뒤 README의 `ARTIFACT_ROOT`, `DATASET_ROOT` 경계를 사용합니다.
3. manifest·seed·overlay provenance는 [`evidence/decision-register.yaml`](../evidence/decision-register.yaml)과 [`environment.lock.json`](../environment.lock.json)을 확인합니다.
4. 공개 주장과 결정 매핑은 `PYTHONPATH=src python3 -m fine_defect_ad.traceability`로 검사합니다.

정확한 재현 범위와 학습/평가 입장 조건은 [sheet-metal 평가](SHEET_METAL_EVALUATION.md), 배포 후보 측정은 [배포 후보 평가](DEPLOYMENT_EVALUATION.md), 공개 의존성·라이선스는 [LICENSES.md](../LICENSES.md)를 참조합니다.
