import numpy as np
import pytest
from fine_defect_ad import superadd_posthoc_evaluation as subject

def test_bilinear_historical_full_map_to_split_shape():
    pytest.importorskip("torch")
    raw=np.arange(1056*4224,dtype='<f4').tobytes()
    result=subject._resize_map(raw,[1056,4224])
    assert result.shape == subject.SPLIT_TARGET_SHAPE
    assert result.dtype == np.dtype('<f4')

def test_resize_rejects_invalid_historical_shape():
    with pytest.raises(ValueError):
        subject._resize_map(np.zeros((4,4),dtype='<f4').tobytes(),[1056,4224])

def test_posthoc_constants_do_not_expose_source_paths():
    assert subject.DERIVED_PREFIX.startswith('superadd-')
    assert subject._path_free({'id_sha256':'a'*64,'source_sha256':'b'*64})
