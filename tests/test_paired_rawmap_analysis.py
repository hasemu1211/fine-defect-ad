import json
from pathlib import Path
import numpy as np
import pytest
from fine_defect_ad.paired_rawmap_analysis import SHAPE, _hash, _manifest_rows, _outputs, mask_features, tie_auroc

def test_tie_aware_pixel_auroc_and_empty_class():
    assert tie_auroc(np.array([0.,1.,1.,2.]),np.array([0,1,0,1],bool)) == pytest.approx(.875)
    assert tie_auroc(np.ones(3),np.zeros(3,bool)) is None

def test_mask_features_are_geometry_normalized():
    m=np.zeros((10,20),bool);m[2:5,4:9]=True
    f=mask_features(m)
    assert f['area_fraction']==pytest.approx(15/200)
    assert f['border_distance_normalized'] is not None and f['elongation'] is not None

def test_manifest_rejects_unhashed_or_wrong_shape(tmp_path):
    d={'status':'SPLIT_E2_TEST_PUBLIC_RAW_MAPS','maps':[{}]*114}; p=tmp_path/'x.json';p.write_text(json.dumps(d))
    with pytest.raises(ValueError):_manifest_rows(tmp_path,p,'SPLIT_E2_TEST_PUBLIC_RAW_MAPS','x-')

def test_outputs_are_hash_bound_and_path_free(tmp_path):
    result={'record':{'status':'PAIRED_RAWMAP_DESCRIPTIVE_ANALYSIS','run_id':'r','privacy':'no paths'},'rows':[{'index':0,'id_sha256':'a'*64,'label':'bad'}],'representatives':[]}
    paths=_outputs(tmp_path,'r',result)
    assert all(p.is_file() for p in paths.values())
    assert _hash(paths['json'].read_bytes()) in paths['json'].name
    assert str(tmp_path) not in paths['json'].read_text() and str(tmp_path) not in paths['csv'].read_text()
