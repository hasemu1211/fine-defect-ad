"""Render small, reproducible portfolio evidence from G002 artifacts."""
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
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' 
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
    boxes=[(30,'검증 정상 이미지'),(220,'EfficientAD-S\n70k 학습 체크포인트'),(440,'원시 이상 맵\nSHA-256 계보'),(640,'검증 전용\n임계값 산출물')]
    body='<text x="28" y="30" class="t">이상 탐지 평가 파이프라인</text><text x="28" y="51" class="s">재현 가능한 산출물; 비교 기준 출처 부재로 판정/F1은 차단 상태</text>'
    for x,label in boxes:
        lines=label.split('\n'); body+=f'<rect x="{x}" y="98" width="150" height="90" rx="10" fill="#fff" stroke="#94a3b8"/>'
        body+=''.join(f'<text x="{x+12}" y="{128+i*20}" class="s">{escape(line)}</text>' for i,line in enumerate(lines))
    for x in [180,370,590]: body+=f'<path d="M{x} 143h28" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'
    body+='<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3z" fill="#64748b"/></marker></defs>'
    return _svg(820,230,body)


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
    assets={'training-curve.svg':training_svg(metrics),'geometry-selection.svg':geometry_svg(freeze),'system-architecture.svg':architecture_svg()}
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
