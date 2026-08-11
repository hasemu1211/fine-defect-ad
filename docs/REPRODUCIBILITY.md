# 재현 방법

이 문서는 **저장소만으로 확인할 수 있는 범위**, **별도 데이터·가중치가 필요한 범위**, **각 실행이 남기는 증거**를 구분합니다. 로컬 경로나 비공개 자산은 공개 명령에 포함하지 않습니다.

## 1. 저장소 계약 검증

데이터셋이나 GPU 없이 코드, 문서 링크, 증거 스키마와 회귀 계약을 확인합니다.

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m fine_defect_ad.traceability
```

첫 명령은 추론·평가·복구·개인정보 비노출 계약을 검사합니다. 두 번째 명령은 README의 결정 참조, 14개 필드 결정 레지스터, 코드·테스트·증거의 추적성 매트릭스를 검사합니다.

## 2. 필요한 비공개 입력

라이선스 또는 용량 때문에 다음 항목은 저장소에 포함하지 않습니다.

| 입력 | 용도 |
| --- | --- |
| MVTec AD 2 데이터셋 | 검증·TESTpub 입력과 정답 마스크 |
| EfficientAD 체크포인트·교사 가중치 | 기준선과 E2-Split 추론 |
| DINOv3 가중치 | SuperADD 특징 추출 |
| TensorRT plan | Triton backend 검증 |
| 아티팩트 저장소 | 원시 맵, 매니페스트, 실패·복구 증거 |

환경은 일반화된 경로로 지정합니다.

```bash
export ARTIFACT_ROOT=/path/to/artifacts
export DATASET_ROOT=/path/to/mvtec_ad_2
export LEASE_DIRECTORY="$ARTIFACT_ROOT/gpu-heavy-events"
```

정확한 체크포인트·교사 가중치 인자는 [Sheet-Metal 평가의 재현 절차](SHEET_METAL_EVALUATION.md#재현)에 있습니다.

## 3. 공개 추론 진입점

단일 이미지에서 기본 E2-Split 원시 이상 맵과 표시용 heatmap, 해시 결속 매니페스트를 생성합니다.

```bash
PYTHONPATH=src python3 -m fine_defect_ad.highres_split infer --help
```

TensorRT/Triton 후보 실행과 증거 형식은 다음 CLI에서 확인합니다.

```bash
PYTHONPATH=src python3 -m fine_defect_ad.tensorrt_promotion --help
```

두 CLI는 입력 이미지 원본을 결과 폴더에 복사하지 않습니다. 실패 시 예외 유형과 지문, 입력·체크포인트 해시를 기록하되 예외 본문과 로컬 경로는 공개 증거에 남기지 않습니다.

## 4. 산출물 검증 방식

재현성은 파일명만 같다고 판단하지 않습니다.

1. 체크포인트와 가중치의 SHA-256을 확인합니다.
2. 데이터 매니페스트의 익명 ID·원본·정답 마스크 해시를 확인합니다.
3. 각 원시 이상 맵의 크기, dtype, shape, 콘텐츠 해시를 확인합니다.
4. 평가 결과가 사용한 매니페스트·평가기·코드 revision을 역으로 연결합니다.
5. 기존 해시 이름의 산출물이 있으면 바이트가 정확히 같을 때만 재사용합니다.

공개 근거는 [`evidence/`](../evidence/)에, 실행 중 생성되는 대용량 원시 맵과 모델 파일은 별도 아티팩트 저장소에 둡니다.

## 5. 재현 범위

- **가능**: 코드 회귀 테스트, 결정·증거 스키마 검증, 공개 JSON 지표 확인, 권한 있는 입력을 이용한 동일 CLI 재실행
- **조건부**: GPU·드라이버·Triton 이미지·가중치가 같은 환경에서의 지연 및 수치 parity 재측정
- **불가능**: 공개 저장소만으로 비공개 데이터·가중치·과거 GPU 환경을 완전히 복원

의존성 출처와 라이선스는 [LICENSES.md](../LICENSES.md), 해석상 제한은 [LIMITATIONS.md](LIMITATIONS.md)에서 확인할 수 있습니다.
