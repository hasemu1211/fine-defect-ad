# 한계

- 동일 TESTpub 114개에 대한 단일 고정 비교는 일반화, 통계적 유의성, SOTA, 운영 최종 모델 선정을 증명하지 않습니다.
- AU-PRO@0.05의 로컬 evaluator 기록은 MVTec AD 2 서버와의 동치나 미확정 comparator 동작을 증명하지 않습니다. 자세한 근거는 [`mvtec-metric-protocol.json`](../evidence/mvtec-metric-protocol.json)에 있습니다.
- E2-Split은 legacy E2와 별도 검증 경로이지만, 실제 생산 결함·설비 변화·장기 drift에 대한 성능을 보장하지 않습니다.
- SuperADD/DINOv3는 비교 연구 범위이며 Triton serving, export/parity, bank serialization은 검증되지 않았습니다.
- 시각화는 raw-map 집계와 익명 raw-map·GT-mask 패널만 사용합니다. 원본 제조 이미지와 로컬 경로를 공개하지 않으므로 원자료 수준의 독립 재검토는 지원하지 않습니다.

경계별 근거와 후속 검증 필요사항은 [결정 레지스터](../evidence/decision-register.yaml) 및 [paired raw-map 분석](PAIRED_RAWMAP_ANALYSIS.md)에 연결되어 있습니다.

### SuperADD reconstruction boundary

The immutable FP32 comparison metrics remain the recorded comparison evidence. A validation-only supplement matched the train-bank SHA but not the recorded FP16 parity hash; FP16 is therefore **not admitted**. Validation-only resource values are available, while original TEST peak VRAM/host RSS remain unavailable; no resource or latency-superiority claim is supported.



Validation evidence index: [`3c6d1101332d44ee3c32942a0e92122d9ed46d611aebafde827a6775ae02ad1d`](../evidence/superadd-vits-validation-evidence-index-3c6d1101332d44ee3c32942a0e92122d9ed46d611aebafde827a6775ae02ad1d.json). It supersedes the unavailable-resource wording: validation-only resource values are available; original TEST peak VRAM/RSS remain unavailable; FP16 is not admitted.
