import json
import re
from hashlib import sha256
from pathlib import Path
from fine_defect_ad.portfolio_render import render, serving_evidence_svg

def test_public_assets_have_no_absolute_paths(tmp_path):
    artifacts=tmp_path/'artifacts'; artifacts.mkdir(); run='r';
    (artifacts/f'g002-metrics-{run}.json').write_text(json.dumps([{'step':1,'train_loss':2.0},{'step':2,'train_loss':1.0}]))
    (artifacts/f'g002-calibration-{run}-x.json').write_text(json.dumps({'raw_threshold':.5,'per_image_max_raw_scores':[]}))
    (artifacts/f'g002-e2-pretest-freeze-{run}-x.json').write_text(json.dumps({'decision_id':'DEC-GEO-002','geometry':{'empirical_border':60},'revision':{'initial_border':16,'revised_border':80,'status':'REVISION_UNSTABLE_RETAIN_E1'},'selection':{'selected':'E1','measured_gates':{'E1':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':True},'E2':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':False}}}}))
    (artifacts/f'g002-validation-raw-maps-{run}.json').write_text(json.dumps({'maps':[]}))
    public=tmp_path/'public'; result=render(artifact_root=artifacts,dataset_root=None,run_id=run,public_dir=public,preview_dir=None)
    assert result['selected_geometry']=='E1'
    geometry=(public/'geometry-selection.svg').read_text()
    assert 'height="540"' in geometry and '256×256 기준 경로' in geometry and '시임/원점 안정성: 미통과' in geometry and '분기 분리형 고해상도 타일링 후보' in geometry and 'DEC-SPLIT-003' in geometry
    assert sorted(p.name for p in public.iterdir())==['geometry-selection.svg','serving-evidence.svg','system-architecture.svg','training-curve.svg']
    architecture=(public/'system-architecture.svg').read_text()
    assert 'width="100%" height="650"' in architecture and '고해상도 타일 분할' in architecture and 'TensorRT FP32 plan' in architecture and 'Triton server' in architecture and 'Hann stitch' in architecture and 'E1' not in architecture
    assert architecture.count('marker-end="url(#a)"') == 5
    serving=(public/'serving-evidence.svg').read_text()
    assert '2.4610 s/image → TensorRT FP32 2.1040 s/image' in serving
    assert 'Image AU-ROC  0.734722 → 0.733333 (-0.001389)' in serving
    assert 'AU-PRO@0.05  0.132685 → 0.132769 (+0.000084)' in serving
    assert str(tmp_path) not in ''.join(p.read_text() for p in public.iterdir())


def test_serving_evidence_svg_uses_supplied_measurements():
    evidence = {
        'baseline_latency_seconds': 4.0, 'candidate_latency_seconds': 3.0,
        'image_auroc_baseline': .1, 'image_auroc_candidate': .2,
        'au_pro_baseline': .3, 'au_pro_candidate': .4,
        'parity_images': 7, 'outside_band_flips': 2,
        'gpu_reserved_bytes': 2 * 1024 * 1024, 'image_digest': 'digest',
    }
    svg = serving_evidence_svg(evidence)
    assert '4.0000 s/image → TensorRT FP32 3.0000 s/image' in svg
    assert '25.0% 감소' in svg and '0.100000 → 0.200000 (+0.100000)' in svg
    assert '검증 이미지 7장' in svg and '판정 flip: 2' in svg and '2.0 MiB' in svg

def test_preview_metadata_binds_each_png(tmp_path):
    import numpy as np
    from PIL import Image
    artifacts=tmp_path/'artifacts'; artifacts.mkdir(); data=tmp_path/'data'; run='r'
    (artifacts/f'g002-metrics-{run}.json').write_text(json.dumps([{'step':1,'train_loss':2.0}]))
    rows=[]; scores=[]
    for index in range(3):
        identity=f'validation/good/{index:03}_regular.png'; source=data/identity; source.parent.mkdir(parents=True,exist_ok=True); Image.new('RGB',(8,8),(index,0,0)).save(source)
        source_sha=sha256(source.read_bytes()).hexdigest(); map_sha=f'{index:064x}'; (artifacts/f'g002-validation-raw-{index:02d}-{map_sha}.bin').write_bytes(np.full((1,1,256,256),index,dtype='<f4').tobytes())
        rows.append({'image_identity':identity,'source_sha256':source_sha,'map_sha256':map_sha,'shape':[1,1,256,256]}); scores.append({'image_identity':identity,'max_raw_score':float(index)})
    (artifacts/f'g002-calibration-{run}-x.json').write_text(json.dumps({'raw_threshold':.5,'per_image_max_raw_scores':scores}))
    (artifacts/f'g002-e2-pretest-freeze-{run}-x.json').write_text(json.dumps({'decision_id':'DEC-GEO-002','geometry':{'empirical_border':60},'revision':{'initial_border':16,'revised_border':80,'status':'REVISION_UNSTABLE_RETAIN_E1'},'selection':{'selected':'E1','measured_gates':{'E1':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':True},'E2':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':False}}}}))
    (artifacts/f'g002-validation-raw-maps-{run}.json').write_text(json.dumps({'maps':rows}))
    preview=tmp_path/'preview'; render(artifact_root=artifacts,dataset_root=data,run_id=run,public_dir=tmp_path/'public',preview_dir=preview)
    for row in json.loads((preview/'metadata.json').read_text())['previews']:
        assert row['preview_sha256']==sha256((preview/row['file']).read_bytes()).hexdigest()

def test_candidate_comparison_keeps_latency_scopes_separate():
    svg = (Path(__file__).parents[1] / 'docs/assets/candidate-comparison.svg').read_text()
    assert '직접 비교 불가' in svg
    assert '동일 TESTpub 114' in svg
    assert 'width="640"' in svg
    assert min(map(int, re.findall(r'font-size:(\d+)px', svg))) >= 16


def test_readme_svg_source_fonts_stay_mobile_legible():
    assets=Path(__file__).parents[1]/'docs/assets'
    for name in ('geometry-selection.svg','serving-evidence.svg','system-architecture.svg','training-curve.svg'):
        text=(assets/name).read_text(); sizes=list(map(int,re.findall(r'font-size:(\d+)px',text))); assert min(sizes)>=15
