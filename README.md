# FineDefect AD

**동일 EfficientAD-S Small 체크포인트에서 구현한 E2-Split 고해상도 분할 추론을, 256×256 기준선과 비교·검증한 제조 이미지 이상 탐지 프로젝트입니다.**

![동일 체크포인트 AU-PRO 비교](evidence/g002/metric-comparison.svg)

## 프로젝트 개요

정상 제조 이미지로 학습한 EfficientAD-S Small에서 이상 점수 맵을 생성합니다. **E2-Split — 고해상도 분할 추론**을 주된 공개 추론 경로로 구현하고, 학습 체크포인트·평가 입력·원시 맵을 SHA-256으로 결속했습니다. 분할 경로의 분위수는 검증 데이터로만 동결했습니다.

이 공개 저장소는 학습·평가·재현 경로를 보여주는 포트폴리오이며, 일반 배포용 추론 서비스가 아닙니다.

동일 체크포인트에서 E2-Split은 E1 256×256 기준선보다 높은 local AU-PRO@0.05를 기록했습니다. 입력 경로와 평가 산출물을 재현 가능하게 남겨, 재학습 효과와 추론 경로 효과를 분리해 확인할 수 있습니다.

## 핵심 구현

- **E2-Split — 고해상도 분할 추론**: 원본 해상도를 256×256 타일로 나눠 국소 교사–학생 특징 잔차 맵을 만들고, 표준 256×256 입력에서 전역 오토인코더–학생 잔차 맵을 한 번 계산해 결합합니다. 타일은 128 px stride와 Hann 가중치로 결합합니다.
- **E1 — 256×256 기준 추론**: 입력 전체를 256×256으로 축소해 EfficientAD-S의 비교 기준 이상 맵을 생성합니다.
- **검증 경계**: E2-Split의 분위수는 검증 정상 이미지 19장으로만 계산합니다. TESTpub·TESTpriv·OOD 입력은 학습·보정·후보 구성에 사용하지 않습니다.
- **추적성**: 체크포인트, 원시 맵, 입력 식별자, 실행 산출물을 SHA-256으로 결속해 다른 입력이나 체크포인트의 혼입을 거부합니다.

## 시스템 아키텍처

![시스템 아키텍처](docs/assets/system-architecture.svg)

정상 데이터 학습 → 체크포인트 고정 → E2-Split 검증 원시 맵·분위수 동결 → E2-Split TESTpub 원시 맵 평가 순서입니다. E1은 같은 체크포인트의 256×256 비교 기준선입니다. 최종 배포 판정과 운영 임계값은 이 저장소의 범위에 포함하지 않습니다.

## 검증 결과

| 항목 | 결과 | 해석 |
| --- | --- | --- |
| 학습 모델 | EfficientAD-S Small, 70,000 step | 두 추론 경로가 같은 체크포인트를 사용 |
| 최종 학습 손실 / 처리량 | 2.16434598 / 7.4513 step/s | 최종 학습 이력 |
| 체크포인트 SHA-256 | `9e7a5f567a83f42d…a4154801` | 실행 식별자와 결속 |
| E2-Split 검증 | `READY`, 원시 맵 19개 | 고해상도 분할 추론의 입력·기하·분위수 동결 |
| E2-Split TESTpub AU-PRO@0.05 | 0.13268484492898858 | 같은 체크포인트에서 E1 대비 약 6.45배 |
| E1 검증 | `READY`, 원시 맵 19개 | 256×256 비교 기준선 |
| E1 TESTpub AU-PRO@0.05 | 0.02058176590668011 | 256×256 비교 기준선 |

E2-Split은 **재학습 없이 개선한 고해상도 추론 경로**입니다. 절대 AU-PRO@0.05 값은 낮으므로 모델 성능의 강점이나 배포 승격으로 해석하지 않습니다. 이 결과는 local TESTpub 평가 산출물이며, AD2 서버·리더보드 결과가 아닙니다.

![학습 추이](docs/assets/training-curve.svg)

## 설계 판단과 한계

- E1과 E2-Split은 같은 체크포인트와 같은 TESTpub 입력 식별자를 사용해, 모델 재학습 효과와 추론 경로 효과를 구분했습니다.
- E2-Split의 `READY`는 검증 입력·타일 기하·분위수 동결 통과를 뜻합니다. 최종 모델 선택이나 운영 배포 판정은 아닙니다.
- 이전 **전체 분기 타일링 경로(legacy E2)** 는 별도 `DEC-GEO-002` 기하 실험에서 시임·원점 안정성 기준을 통과하지 못했습니다. 이는 구조적으로 수정된 E2-Split의 검증 결과가 아니며, 상세 증거는 [G002 평가와 한계](docs/G002_EVALUATION.md)에 남겼습니다.

## 빠른 실행

```bash
python3 -m pytest -q

export ARTIFACT_ROOT=/path/to/artifacts
export DATASET_ROOT=/path/to/dataset
PYTHONPATH=src python3 -m fine_defect_ad.highres_split infer --help
PYTHONPATH=src python3 -m fine_defect_ad.evaluation_history --help
```

E2-Split이 기본 단일 이미지 원시 맵 경로입니다(임계값·판정 없음).

```bash
PYTHONPATH=src python3 -m fine_defect_ad.highres_split infer \
  --artifact-root "$ARTIFACT_ROOT" --checkpoint "$CHECKPOINT" --metrics "$METRICS" \
  --final-attempt "$FINAL_ATTEMPT" --training-identity "$TRAINING_IDENTITY" \
  --dataset-root "$DATASET_ROOT" --teacher-small "$TEACHER_SMALL" --imagenette-root "$IMAGENETTE_ROOT" \
  --lease-directory "$LEASE_DIRECTORY" --run-id "$RUN_ID" --input-image /path/to/inspection.png \
  --split-freeze "$PRETEST_FREEZE"
```

`--mode e1`은 같은 체크포인트의 256×256 비교 기준선이며 `--split-freeze`를 받지 않습니다. 실제 실행에는 학습 체크포인트, teacher 가중치, Imagenette, GPU lease 디렉터리가 필요합니다. 상세 재현은 [평가 문서](docs/G002_EVALUATION.md#재현)를 따릅니다.

## 저장소 구조

```text
src/fine_defect_ad/  학습, 추론 경로, 검증, 평가 런타임
tests/               단위·통합 회귀 테스트
docs/                공개 포트폴리오 문서와 도식
evidence/            공개 가능한 평가 기준선·비교·입력 해시
```

## 상세 문서

- [G002 평가와 한계](docs/G002_EVALUATION.md)
- [기준선 평가](evidence/g002/baseline_evaluation.json)
- [평가 비교](evidence/g002/evaluation_comparison.json)
- [검토 우선 사례](evidence/g002/error_cases.csv)
- [라이선스와 출처](LICENSES.md)
