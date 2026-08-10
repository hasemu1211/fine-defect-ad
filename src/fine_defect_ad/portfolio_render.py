"""Render small, reproducible portfolio evidence from recorded artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from xml.sax.saxutils import escape


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def _svg(width: int, height: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{height}" viewBox="0 0 {width} {height}">'
            '<style>text{font-family:system-ui,sans-serif;fill:#172033}.t{font-size:18px;font-weight:700}.s{font-size:12px;fill:#526070}.n{font-size:11px;fill:#526070}</style>'
            f'<rect width="100%" height="100%" fill="#f8fafc"/>{body}</svg>')


def training_svg(metrics: list[dict]) -> str:
    rows = [r for r in metrics if 'step' in r and 'train_loss' in r]
    if not rows:
        raise ValueError('metrics has no train_loss rows')
    w, h, left, bottom = 720, 300, 54, 246
    mx = max(r['step'] for r in rows); lo = min(r['train_loss'] for r in rows); hi = max(r['train_loss'] for r in rows)
    span = max(hi - lo, 1e-12)
    points = ' '.join(f"{left+(w-left-24)*r['step']/mx:.1f},{bottom-(bottom-48)*(r['train_loss']-lo)/span:.1f}" for r in rows)
    return _svg(w,h, f'<text x="28" y="30" class="t">EfficientAD-S 학습 손실</text><text x="28" y="50" class="s">실행 증거 · 에폭 스냅샷 {len(rows)}개 · 단계 {mx}</text><line x1="{left}" y1="48" x2="{left}" y2="{bottom}" stroke="#cbd5e1"/><line x1="{left}" y1="{bottom}" x2="696" y2="{bottom}" stroke="#cbd5e1"/><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/><text x="{left}" y="270" class="n">0</text><text x="650" y="270" class="n">{mx}</text><text x="{left}" y="72" class="n">손실 {hi:.3f}</text><text x="{left}" y="238" class="n">손실 {lo:.3f}</text>')


def geometry_svg(freeze: dict) -> str:
    rev = freeze['revision']; selection = freeze['selection']; e2=selection['measured_gates']['E2']; empirical=freeze['geometry']['empirical_border']
    seam_status = '통과' if e2['origin_seam_valid'] else '미통과'
    return _svg(820,330, f'<text x="28" y="30" class="t">타일 경계 처리 결정: 256×256 기준 경로 유지</text><text x="28" y="52" class="s">검증 정상 이미지 쌍 비교 · 결정 {escape(freeze["decision_id"])}</text><rect x="42" y="82" width="290" height="160" rx="10" fill="#dcfce7" stroke="#16a34a"/><text x="62" y="112" class="t">256×256 기준 경로</text><text x="62" y="140" class="s">내부 식별자: E1</text><text x="62" y="164" class="s">범위·무결성·시임 게이트 통과</text><text x="62" y="188" class="s">결정: 기준 경로 유지</text><rect x="488" y="82" width="290" height="160" rx="10" fill="#fee2e2" stroke="#dc2626"/><text x="508" y="112" class="t">전체 분기 타일링 경로</text><text x="508" y="140" class="s">legacy E2</text><text x="508" y="164" class="s">border 기록: {rev["initial_border"]} → {rev["revised_border"]} → {empirical} px</text><text x="508" y="188" class="s">시임/원점 안정성: {seam_status} · 미채택</text><text x="382" y="164" class="n">쌍 비교</text><text x="28" y="282" class="n">DEC-GEO-002는 후속 분기 분리형 고해상도 타일링 후보(DEC-SPLIT-003)와 별개의 기하 결정입니다.</text><text x="28" y="304" class="n">이 결정에는 TESTpub·OOD 입력을 사용하지 않았습니다.</text>')


def architecture_svg() -> str:
    arrow = '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3z" fill="#64748b"/></marker></defs>'
    def box(x, y, w, h, fill, title, detail):
        rect = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#64748b"/>'
        if not detail:
            return rect + f'<text x="{x+w/2}" y="{y+h/2+6}" class="t" text-anchor="middle">{title}</text>'
        return rect + f'<text x="{x+20}" y="{y+38}" class="t">{title}</text><text x="{x+20}" y="{y+67}" class="s">{detail}</text>'
    body = ('<text x="48" y="48" class="t">고해상도 분할 추론과 TensorRT FP32 서빙 경로</text>'
            '<text x="48" y="74" class="s">같은 EfficientAD-S 체크포인트 · TensorRT FP32 plan · Triton HTTP binary transport</text>'
            + box(400, 105, 320, 72, '#ffffff', '고해상도 입력 이미지', '원본 해상도 유지')
            + box(400, 215, 320, 86, '#e0f2fe', '고해상도 타일 분할', '256×256 타일 · 고정 기하')
            + box(125, 355, 355, 90, '#ede9fe', 'TensorRT FP32 plan', '고정 입력 · 배치 4')
            + box(640, 355, 355, 90, '#fff7ed', 'Triton server', 'plan load · HTTP binary transport')
            + box(335, 470, 450, 86, '#dcfce7', 'stitch · 맵 결합', 'Hann stitch · 동결 분위수 정규화')
            + box(335, 585, 450, 38, '#ffffff', '원시 이상 맵 → 검증 / TESTpub 평가', '')
            + '<path d="M560 177v31" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
            + '<path d="M560 301v25H302v22" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
            + '<path d="M480 400h153" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
            + '<path d="M818 445v12H760v6" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
            + '<path d="M560 556v22" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
            + arrow)
    return _svg(1120, 640, body)


# Public labels are deliberately narrow summaries of the immutable run evidence.
# They contain no host paths or dataset images.
SERVING_EVIDENCE = {
    "baseline_latency_seconds": 2.4610,
    "candidate_latency_seconds": 2.1040,
    "image_auroc_baseline": 0.734722,
    "image_auroc_candidate": 0.733333,
    "au_pro_baseline": 0.132685,
    "au_pro_candidate": 0.132769,
    "parity_images": 3,
    "outside_band_flips": 0,
    "gpu_reserved_bytes": 465_567_744,
    "image_digest": "a40838bb4587d2aceb46b1e7fd144afb24c9016c219dd3eba31716e4e28dbfc7",
}


def serving_evidence_svg(evidence: dict = SERVING_EVIDENCE) -> str:
    """Render compact, value-labelled serving and backend-A/B evidence."""
    baseline = float(evidence["baseline_latency_seconds"])
    candidate = float(evidence["candidate_latency_seconds"])
    reduction = (1 - candidate / baseline) * 100
    auroc_delta = float(evidence["image_auroc_candidate"]) - float(evidence["image_auroc_baseline"])
    aupro_delta = float(evidence["au_pro_candidate"]) - float(evidence["au_pro_baseline"])
    reserved_mib = int(evidence["gpu_reserved_bytes"]) / 1024 / 1024
    digest = str(evidence["image_digest"])
    digest_summary = escape(f"{digest[:8]}…{digest[-8:]}") if len(digest) > 16 else escape(digest)
    body = (
        '<text x="48" y="48" class="t">TensorRT FP32 + Triton: 측정 요약</text>'
        '<text x="48" y="74" class="s">대표 단일 이미지 지연과 114-image backend A/B 평가는 서로 다른 측정입니다.</text>'
        '<rect x="48" y="104" width="1024" height="112" rx="12" fill="#e0f2fe" stroke="#64748b"/>'
        '<text x="72" y="138" class="t">대표 고해상도 E2E 지연</text>'
        f'<text x="72" y="168" class="s">TorchScript B4 {baseline:.4f} s/image → TensorRT FP32 {candidate:.4f} s/image</text>'
        f'<text x="72" y="196" class="t">{reduction:.1f}% 감소</text>'
        '<text x="438" y="138" class="s">단일 대표 이미지 · 256 tiles · 서버 준비 후 측정</text>'
        '<text x="438" y="168" class="s">실시간 처리량 또는 운영 SLA 주장이 아님</text>'
        '<rect x="48" y="242" width="498" height="184" rx="12" fill="#f8fafc" stroke="#64748b"/>'
        '<text x="72" y="276" class="t">TESTpub backend A/B · 114 images</text>'
        '<text x="72" y="306" class="s">동일 체크포인트 · 재보정/튜닝 없음</text>'
        f'<text x="72" y="340" class="s">Image AU-ROC  {float(evidence["image_auroc_baseline"]):.6f} → {float(evidence["image_auroc_candidate"]):.6f} ({auroc_delta:+.6f})</text>'
        f'<text x="72" y="372" class="s">AU-PRO@0.05  {float(evidence["au_pro_baseline"]):.6f} → {float(evidence["au_pro_candidate"]):.6f} ({aupro_delta:+.6f})</text>'
        '<text x="72" y="402" class="n">총 evaluator 시간은 114장 전체 실행 시간이며, 대표 지연과 직접 비교하지 않습니다.</text>'
        '<rect x="574" y="242" width="498" height="184" rx="12" fill="#f8fafc" stroke="#64748b"/>'
        '<text x="598" y="276" class="t">검증·실행 결속</text>'
        f'<text x="598" y="306" class="s">검증 이미지 {int(evidence["parity_images"])}장: 이미지 판정 동일</text>'
        f'<text x="598" y="338" class="s">측정 불확실성 대역 밖 판정 flip: {int(evidence["outside_band_flips"])}</text>'
        f'<text x="598" y="370" class="s">서버 준비 후 GPU reserved peak: {reserved_mib:.1f} MiB</text>'
        f'<text x="598" y="402" class="n">고정 Triton image digest: sha256:{digest_summary}</text>'
        '<text x="560" y="460" class="n" text-anchor="middle">후보 백엔드의 수치 보존과 추적성 검증 결과이며, production promotion 결론은 아닙니다.</text>'
    )
    return _svg(1120, 490, body)


def _find_source(root: Path, identity: str, digest: str) -> Path:
    direct=root/identity
    candidates=[direct] if direct.is_file() else list(root.rglob(Path(identity).name))
    for path in candidates:
        if path.is_file() and _sha(path)==digest:
            return path
    raise FileNotFoundError(f'cannot find source matching {identity} under supplied dataset root')


def _preview(entries: list[dict], artifact_root: Path, dataset_root: Path, output: Path, run_id: str, threshold: float) -> dict:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('preview rendering requires installed numpy and Pillow') from exc
    output.mkdir(parents=True, exist_ok=True)
    selected=sorted(entries,key=lambda x:(x['max_raw_score'],x['image_identity']))
    selected=[selected[0],selected[len(selected)//2],selected[-1]]
    result=[]
    for rank,row in zip(('low','median','high'),selected):
        source=_find_source(dataset_root,row['image_identity'],row['source_sha256'])
        raw=artifact_root/f"g002-validation-raw-{int(row['image_identity'].split('/')[-1].split('_')[0]):02d}-{row['map_sha256']}.bin"
        shape=tuple(row['shape']); heat=np.fromfile(raw,dtype='<f4').reshape(shape)[0,0]
        image=Image.open(source).convert('RGB').resize((256,256))
        norm=(255*(heat-heat.min())/max(float(heat.max()-heat.min()),1e-12)).astype('uint8')
        red=np.zeros((256,256,3),dtype='uint8'); red[:,:,0]=norm; red[:,:,1]=(norm*.15).astype('uint8')
        heatmap=Image.fromarray(red); overlay=Image.blend(image,heatmap,.45)
        canvas=Image.new('RGB',(768,256)); canvas.paste(image,(0,0)); canvas.paste(heatmap,(256,0)); canvas.paste(overlay,(512,0))
        name=f'{rank}-{Path(row["image_identity"]).stem}.png'; canvas.save(output/name)
        result.append({'tag':rank,'file':name,'preview_sha256':_sha(output/name),'image_identity':row['image_identity'],'source_sha256':row['source_sha256'],'map_sha256':row['map_sha256'],'max_raw_score':row['max_raw_score'],'threshold':threshold,'split':'validation-good','selected_geometry':'E1','model':'EfficientAD-S','run_id':run_id,'redistribution':'PUBLIC_REDISTRIBUTION_NOT_VERIFIED'})
    metadata={'schema':'portfolio-preview/v1','redistribution':'PUBLIC_REDISTRIBUTION_NOT_VERIFIED','previews':result}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def render(*, artifact_root: Path, dataset_root: Path | None, run_id: str, public_dir: Path, preview_dir: Path | None) -> dict:
    metrics=_read(artifact_root/f'g002-metrics-{run_id}.json')
    calibration=next(( _read(p) for p in artifact_root.glob(f'g002-calibration-{run_id}-*.json')),None)
    freeze=next((_read(p) for p in artifact_root.glob(f'g002-e2-pretest-freeze-{run_id}-*.json')),None)
    manifest=_read(artifact_root/f'g002-validation-raw-maps-{run_id}.json')
    if not calibration or not freeze: raise FileNotFoundError('calibration and frozen geometry artifacts are required')
    public_dir.mkdir(parents=True,exist_ok=True)
    assets={'training-curve.svg':training_svg(metrics),'geometry-selection.svg':geometry_svg(freeze),
            'system-architecture.svg':architecture_svg(),'serving-evidence.svg':serving_evidence_svg()}
    for name,text in assets.items(): (public_dir/name).write_text(text+'\n')
    scores={r['image_identity']:r for r in calibration['per_image_max_raw_scores']}
    entries=[{**m,**scores[m['image_identity']]} for m in manifest['maps']]
    threshold=calibration['raw_threshold']
    out={'public_assets':sorted(assets),'run_id':run_id,'model':'EfficientAD-S','selected_geometry':freeze['selection']['selected'],'threshold':threshold}
    if preview_dir is not None:
        if dataset_root is None: raise ValueError('dataset root is required for preview rendering')
        out['preview']=_preview(entries,artifact_root,dataset_root,preview_dir,run_id,threshold)
    return out


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True); p.add_argument('--dataset-root',type=Path)
    p.add_argument('--run-id',required=True); p.add_argument('--public-dir',type=Path,default=Path('docs/assets'))
    p.add_argument('--preview-dir',type=Path)
    args=p.parse_args()
    print(json.dumps(render(artifact_root=args.artifact_root,dataset_root=args.dataset_root,run_id=args.run_id,public_dir=args.public_dir,preview_dir=args.preview_dir),indent=2))

if __name__=='__main__': main()


def candidate_comparison_svg() -> str:
    """Raw-map-only SuperADD comparison; latency scopes are intentionally separate."""
    return _svg(1120, 180, '<text x="48" y="48" class="t">SuperADD evidence candidate</text><text x="48" y="78" class="s">Image AU-ROC 0.83935185 · AU-PRO@0.05 0.43140701</text><text x="48" y="108" class="s">SuperADD 114-image inference: mean 1.1414 s, p50 1.0242 s; E2-Split representative single image: 2.1040 s</text><text x="48" y="140" class="n">Latency scopes are not directly comparable. Serving NO-GO: export, feature/final-map parity, and bank serialization proof absent.</text>')
