# G002 평가

## 범위

이 문서는 EfficientAD-S Small의 학습 산출물과 검증 전용 평가·기하 선택·원시 임계값 보정 결과를 기록합니다. 최종 성능 평가나 운영 배포 판정 문서가 아닙니다.

## 핵심 지표

| 구분 | 결과 | 근거/상태 |
| --- | --- | --- |
| 학습 | 70,000 step, 최종 loss 2.1643459796905518 | 학습 이력 저장 |
| 학습 처리량 | 7.4513 step/s | 최종 학습 이력 |
| 모델 체크포인트 | SHA-256 `9e7a5f567a83f42dacf80318df3d3bd33b7a7c922b1035bb529ecf59a4154801` | 실행 식별자와 결속 |
| E1 원시 맵 | 19개, `READY` | 검증 정상 이미지 전용 |
| E2 원시 맵 | 19개, `READY`, 241.286 s | 전체 해상도 타일링 경로 |
| E2 GPU 메모리 | allocated 105,738,752 B / reserved 148,897,792 B | 실행 증거 |
| E1 TESTpub 원시 맵 | 114개(정상 24 / 불량 90), `READY`, 20.523 s | 최초 평가 기준선 |
| Split E2 TESTpub 원시 맵 | 114개(정상 24 / 불량 90), `READY` | 동일 체크포인트의 고해상도 결합 경로 |
| E1 TESTpub AU-PRO@0.05 | 0.02058176590668011 | 최초 평가 기준선 |
| Split E2 TESTpub AU-PRO@0.05 | 0.13268484492898858 | 파이프라인 변경 결과, AD2 서버·리더보드 결과 아님 |
| 기하 선택 | E1 | E2 시임 검증 불안정 |
| 보정 표본 | 1,245,184 pixel | 선택된 E1 원시 맵 |
| 평균 / 모집단 표준편차 | 0.07112031954392078 / 0.04543306480528533 | 검증 전용 점수 |
| 원시 임계값 | 0.20741951395977676 | `mean + 3 × population std` |

## 증거와 추적성

선택된 체크포인트, sidecar, 학습 이력, 최종 실행, 학습 식별자는 SHA-256으로 결속됩니다. 검증 런타임은 동일한 런 ID와 이 입력 결속을 확인한 뒤 원시 맵·매니페스트·실행 증거를 기록합니다.

기하 선택은 `DEC-GEO-002` 동결 증거에 기록됩니다. E1과 E2를 검증 데이터에서만 비교하고, 동결된 선택과 매니페스트가 일치할 때만 보정이 진행됩니다. 보정 산출물은 선택된 E1 맵과 별도의 post-selection 결속을 기록합니다.

TESTpub은 선택된 E1 경로에서 원시 맵만 추출했고, 해당 단계의 실행 증거 SHA-256은 `02829a3ddfc0e9879ba33e7cde4ce5f9a3e50ed0a2e6f7a0abf07f22961e8523`입니다.

Split E2 AU-PRO@0.05의 증거 SHA-256은 `1be1dc9be407f5c0efeed526a3763f98ad08a1bd1aeb8c806f6282435022b0b2`입니다. `command=g002-eval-test-public-e2-split-au-pro-0.05`, `protocol=POST_HOC_PIPELINE_CORRECTION__ONE_SHOT_TESTPUB__NO_TUNING`으로 기록됩니다. 최초 실행의 평가 키 오류 로그 SHA-256 `2a5d62d01d60ef825640335885064e604b094dfa5b55d0ffd5814834b0c925b9`와 수정 커밋 `278722b`을 복구 증거에 결속했으며, 모델을 다시 학습하지 않았습니다.

[MVTec AD 2 Code Utils](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)는 정량적 TESTpub AU-PRO를 [MVTec AD evaluator v1.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)으로 평가하도록 안내합니다. evaluator archive SHA-256 `dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220`를 검증한 뒤 고정된 원시 맵을 평가했습니다. E1 기준선 증거 SHA-256은 `bfd104d65d79946b4aab9620e5b975d86061cc1cff255eb6f98daf41bae52a56`, Split E2 증거 SHA-256은 `1be1dc9be407f5c0efeed526a3763f98ad08a1bd1aeb8c806f6282435022b0b2`입니다.

## 최초 평가 추적

- [기준선 평가](../evidence/g002/baseline_evaluation.json): 최초 E1 지표와 입력 증거 해시
- [평가 비교](../evidence/g002/evaluation_comparison.json): 동일 체크포인트에서 파이프라인 변경 전후의 절대·상대 변화
- [검토 우선 사례](../evidence/g002/error_cases.csv): 정상 고점수·불량 저점수 순위와 맵 통계

FP/FN은 동결된 판정 임계값이 없으므로 계산하지 않습니다. 오류 사례 파일은 모델 판정 결과가 아니라 후속 분석 우선순위입니다.

## 선택 논리

E2는 고해상도 타일링 경로로 map border 80 px를 시도한 뒤 두 번째 경험적 border 60 px로 재측정했습니다. 재측정 결과는 `REVISION_UNSTABLE_RETAIN_E1` 상태였고, E1/E2의 경계·시임 응답을 계층적 비가중 규칙으로 검사한 결과 E2의 시임 검증은 안정 기준을 만족하지 못했습니다. 따라서 동결된 선택은 **E1**입니다. 이 선택은 성능 우위 주장이 아니라 검증 규칙에 따른 입력 경로 선택입니다.

## 한계

- TESTpub은 학습·보정에 사용하지 않았고, local AU-PRO@0.05는 고정된 평가 산출 체인으로만 기록했습니다.
- local AU-PRO@0.05 `0.13268484492898858`은 낮은 수치입니다. 재현 가능한 파이프라인과 평가 입력 결속의 증거이지 모델 성능의 강점으로 해석하지 않습니다.
- `0.02058 → 0.13268` 변화는 동일 체크포인트의 파이프라인 변경 결과이며 모델 재학습 성능 향상으로 해석하지 않습니다.
- E1 보정 입력은 paired E1 probe 맵과 canonical 맵이 바이트 단위로 동일하지 않은 수치 전처리 변형을 포함합니다. 관측된 최대 입력 차이는 `1.1920929e-7`이며, 해당 차이는 공개 성능 주장으로 해석하지 않습니다.
- 공개 SVG는 집계 수치와 구조만 보여 줍니다. 실제 데이터셋 기반 미리보기는 재배포 권한 확인 전까지 로컬 증거로 유지합니다.

## 재현

아래 변수는 일반화된 저장 위치를 사용합니다. 명령은 실제 CLI 인자와 일치하며, GPU가 사용 가능한 Python 환경에서 실행해야 합니다.

두 평가 증거와 원시 맵 매니페스트가 있으면 다음 재사용 CLI로 기준선·비교·검토 우선 사례를 생성할 수 있습니다.

```bash
PYTHONPATH=src python3 -m fine_defect_ad.evaluation_history \
  --baseline-evidence "$BASELINE_EVIDENCE" --candidate-evidence "$CANDIDATE_EVIDENCE" \
  --baseline-manifest "$BASELINE_MANIFEST" --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-binding "$CANDIDATE_BINDING" \
  --artifact-root "$ARTIFACT_ROOT" --output-dir evidence/g002 \
  --metric-path local_au_pro.output.au_pro_0_05 \
  --change-scope pipeline_only_same_checkpoint \
  --change-description "동일 체크포인트의 추론 파이프라인 변경" \
  --normal-label good --anomaly-label bad \
  --baseline-label "E1 256×256" --candidate-label "Split E2 고해상도"
```

```bash
export RUN_ID=r1-efficientad-full-20260804a
export ARTIFACT_ROOT=/path/to/artifacts
export DATASET_ROOT=/path/to/mvtec_ad_2
export TEACHER_SMALL=/path/to/pretrained_teacher_small.pth
export IMAGENETTE_ROOT=/path/to/imagenette2
export LEASE_DIRECTORY="$ARTIFACT_ROOT/gpu-heavy-events"
export CHECKPOINT="$ARTIFACT_ROOT/g002-last-$RUN_ID-1.ckpt"
export SIDECAR="$CHECKPOINT.json"
export METRICS="$ARTIFACT_ROOT/g002-metrics-$RUN_ID.json"
export FINAL_ATTEMPT=/path/to/final-attempt.json
export TRAINING_IDENTITY=/path/to/training-identity.json
export E1_MANIFEST="$ARTIFACT_ROOT/g002-validation-raw-maps-$RUN_ID.json"
export PRETEST_FREEZE_SHA256=a2edec773c773d5a006b87da81633f790b68b33de3050044e7bb0cd0d2a25d67
export PRETEST_FREEZE="$ARTIFACT_ROOT/g002-e2-pretest-freeze-$RUN_ID-$PRETEST_FREEZE_SHA256.json"
export GEOMETRY_EVIDENCE="$PRETEST_FREEZE"
export GEOMETRY_EVIDENCE_SHA256="$(sha256sum "$GEOMETRY_EVIDENCE" | awk '{print $1}')"
export GEOMETRY_DECISION_ID=DEC-GEO-002
```

E1 검증 맵을 생성합니다.

```bash
PYTHONPATH=src python3 -m fine_defect_ad.g002_eval_runtime \
  --artifact-root "$ARTIFACT_ROOT" --checkpoint "$CHECKPOINT" --sidecar "$SIDECAR" \
  --metrics "$METRICS" --final-attempt "$FINAL_ATTEMPT" \
  --training-identity "$TRAINING_IDENTITY" --dataset-root "$DATASET_ROOT" \
  --teacher-small "$TEACHER_SMALL" --imagenette-root "$IMAGENETTE_ROOT" \
  --lease-directory "$LEASE_DIRECTORY" --run-id "$RUN_ID"
```

E2 원시 맵과 기하 동결 증거를 생성합니다.

```bash
PYTHONPATH=src python3 -m fine_defect_ad.g002_e2_runtime \
  --artifact-root "$ARTIFACT_ROOT" --checkpoint "$CHECKPOINT" --sidecar "$SIDECAR" \
  --metrics "$METRICS" --final-attempt "$FINAL_ATTEMPT" \
  --training-identity "$TRAINING_IDENTITY" --dataset-root "$DATASET_ROOT" \
  --teacher-small "$TEACHER_SMALL" --imagenette-root "$IMAGENETTE_ROOT" \
  --lease-directory "$LEASE_DIRECTORY" --run-id "$RUN_ID"
```

보정은 E2 동결 증거가 선택한 E1 매니페스트와 동일한 기하 증거를 사용해야 합니다. 다음 Bash 명령은 먼저 CPU 전용 post-selection 결속을 생성한 뒤, 그 결속으로 원시 임계값을 보정합니다.

```bash
CALIBRATION_ARGS=(
  --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID"
  --raw-map-manifest "$E1_MANIFEST" --training-identity "$TRAINING_IDENTITY"
  --checkpoint "$CHECKPOINT" --sidecar "$SIDECAR" --metrics "$METRICS"
  --final-attempt "$FINAL_ATTEMPT" --dataset-root "$DATASET_ROOT"
  --geometry-evidence "$GEOMETRY_EVIDENCE"
  --geometry-evidence-sha256 "$GEOMETRY_EVIDENCE_SHA256"
  --geometry-decision-id "$GEOMETRY_DECISION_ID" --pretest-freeze "$PRETEST_FREEZE"
)
BINDING_JSON="$(PYTHONPATH=src python3 -m fine_defect_ad.g002_calibration \
  "${CALIBRATION_ARGS[@]}" --build-post-selection-binding)"
export POST_SELECTION_BINDING="$(printf '%s' "$BINDING_JSON" | \
  python3 -c 'import json, sys; print(json.load(sys.stdin)["artifact"])')"
PYTHONPATH=src python3 -m fine_defect_ad.g002_calibration \
  "${CALIBRATION_ARGS[@]}" --post-selection-binding "$POST_SELECTION_BINDING"
```
