# 아키텍처

이 포트폴리오는 하나의 운영 시스템 설계가 아니라, 동일한 공개 비교 경계에서 두 독립 구현을 검증한 기록입니다.

- **EfficientAD E2-Split + TensorRT/Triton**: 로컬 타일 residual과 전역 residual을 결합하는 고해상도 후보 경로입니다. legacy E2와 분리되어 있으며, 결정과 제한은 [sheet-metal 평가](SHEET_METAL_EVALUATION.md)에 있습니다.
- **SuperADD/DINOv3**: pinned ViT-S 기반의 독립 비교 경로입니다. 현재 Triton serving 범위에는 포함되지 않습니다.
- **공통 경계**: 익명 TESTpub ID, 528×2112 raw-map geometry, 해시 결속, 기록된 evaluator입니다. 이는 운영 최종 모델 선정이나 일반화 우열 판단이 아닙니다.

시스템 흐름과 serving 후보 근거는 [배포 후보 평가](DEPLOYMENT_EVALUATION.md), 비교의 원시 맵 재생 경계는 [paired raw-map 분석](PAIRED_RAWMAP_ANALYSIS.md), 결정별 근거는 [`evidence/decision-register.yaml`](../evidence/decision-register.yaml)을 참조합니다.
