# 평가

공개 비교는 동일 TESTpub 114개 익명 ID의 연속 raw anomaly map을 한 번 평가한 기록입니다. 재학습, TEST tuning, threshold 기반 판정, 운영 모델 선택은 포함하지 않습니다.

- 지표: Image AU-ROC와 AU-PRO@0.05. 로컬 evaluator의 출처·미확인 범위는 [`mvtec-metric-protocol.json`](../evidence/mvtec-metric-protocol.json)에 기록됩니다.
- 고해상도 경로·legacy E2와 E2-Split의 구분: [sheet-metal 평가](SHEET_METAL_EVALUATION.md)
- 동결 raw-map의 tie-aware pixel ranking 분석과 한계: [paired raw-map 분석](PAIRED_RAWMAP_ANALYSIS.md)
- 수치 산출물: [기준선](../evidence/g002/baseline_evaluation.json), [경로 비교](../evidence/g002/evaluation_comparison.json)

서로 다른 evidence scope의 지연 수치는 속도 우위로 해석하지 않습니다.

- SuperADD validation evidence index: [final evidence index](../evidence/superadd-vits-validation-evidence-index-3c6d1101332d44ee3c32942a0e92122d9ed46d611aebafde827a6775ae02ad1d.json)
