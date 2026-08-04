import pytest
from fine_defect_ad.geometry import *
def test_resize_and_keyword_only_tiles():
 assert global_resize((3,4),(6,8)).target==(6,8)
 with pytest.raises(TypeError): bounded_tiles((5,7),(3,4),(1,1))
 plan,boxes=bounded_tiles((5,7),(3,4),invalid_border=(0,0));assert boxes[-1]==(2,3,5,7) and plan.overlap==(0,0)
def test_stitch_masks_borders_and_rejects_arbitrary_assembly():
 plan,boxes=bounded_tiles((3,3),(2,2),invalid_border=(0,0));tiles=[(box,[[y+i+x+j for j in range(2)]for i in range(2)]) for y,x,_,_ in boxes for box in [(y,x,y+2,x+2)]];assert stitch((3,3),tiles,plan=plan,boxes=boxes)[2][2]==4
 with pytest.raises(ValueError):stitch((3,3),[((0,0,1,1),[[1]])],plan=plan,boxes=boxes)
 good,bad=synthetic_diagnostics(),synthetic_diagnostics(corrupt=True)
 assert all(v['max_reconstruction_error']==0 and v['max_seam_band_error']==0 for v in good['patterns'].values())
 assert all(v['max_seam_band_error']>0 for v in bad['patterns'].values()) and good['coverage_multiplicity']['min']>=1
def rows(distance=20, offset=4):
 result=[]
 for kind in ('normal','probe_delta'):
  for origin in ((0,0),(0,0),(offset,offset),(offset,offset)):
   result.append({'image_identity':'validation/good/a','pixel':(distance,distance),'origin':origin,'tile_shape':(256,256),'score':.5,'kind':kind})
 return result
def test_empirical_plan_is_complete_deterministic_and_validation_only():
 kwargs={'approved_validation_identities':[('validation/good/a','hash')]};one=empirical_border_distance_diagnostic(rows(20),**kwargs);two=empirical_border_distance_diagnostic(rows(20),**kwargs)
 assert one['invalid_border']==(16,16) and one['stride']==(224,224) and one['overlap']==(32,32) and one['repeatability_evidence_sha256']==two['repeatability_evidence_sha256']
 plan=empirical_border_distance_diagnostic(rows(28),**kwargs);assert plan['invalid_border']==(24,24) and plan['stride']==(208,208) and plan['overlap']==(48,48)
 with pytest.raises(ValueError):empirical_border_distance_diagnostic(rows(128,2),**kwargs)
 for forged in ('TESTpub/a','validation/good/../TESTpriv/a','validation/good/a_ood'):
  with pytest.raises(ValueError):empirical_border_distance_diagnostic([dict(rows()[0],image_identity=forged),*rows()[1:]],**kwargs)
 with pytest.raises(ValueError):empirical_border_distance_diagnostic([dict(rows()[0],kind='fake'),*rows()[1:]],**kwargs)
