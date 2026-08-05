import json
from pathlib import Path
from fine_defect_ad.portfolio_render import render

def test_public_assets_have_no_absolute_paths(tmp_path):
    artifacts=tmp_path/'artifacts'; artifacts.mkdir(); run='r';
    (artifacts/f'g002-metrics-{run}.json').write_text(json.dumps([{'step':1,'train_loss':2.0},{'step':2,'train_loss':1.0}]))
    (artifacts/f'g002-calibration-{run}-x.json').write_text(json.dumps({'raw_threshold':.5,'per_image_max_raw_scores':[]}))
    (artifacts/f'g002-e2-pretest-freeze-{run}-x.json').write_text(json.dumps({'decision_id':'DEC-GEO-002','revision':{'initial_border':16,'revised_border':80},'selection':{'selected':'E1','measured_gates':{'E1':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':True},'E2':{'coverage_valid':True,'integrity_valid':True,'origin_seam_valid':False}}}}))
    (artifacts/f'g002-validation-raw-maps-{run}.json').write_text(json.dumps({'maps':[]}))
    public=tmp_path/'public'; result=render(artifact_root=artifacts,dataset_root=None,run_id=run,public_dir=public,preview_dir=None)
    assert result['selected_geometry']=='E1'
    assert sorted(p.name for p in public.iterdir())==['geometry-selection.svg','system-architecture.svg','training-curve.svg']
    assert str(tmp_path) not in ''.join(p.read_text() for p in public.iterdir())
