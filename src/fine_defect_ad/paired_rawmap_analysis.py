"""Frozen, privacy-safe paired analysis of E2-Split and SuperADD raw maps.

This command only reads already persisted TESTpub maps plus masks; it never loads a
model, decodes source images, tunes a threshold, or changes either candidate.
"""
from __future__ import annotations
import argparse, csv, json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

SHAPE=(528,2112); N=114

def _canon(x: Any)->bytes: return json.dumps(x, sort_keys=True, separators=(',',':'), allow_nan=False).encode()
def _hash(b: bytes)->str: return sha256(b).hexdigest()
def _read(p: Path)->dict[str,Any]:
    try: return json.loads(p.read_text())
    except Exception as e: raise ValueError('invalid manifest') from e

def tie_auroc(scores: np.ndarray, positive: np.ndarray)->float|None:
    """Exact tie-aware Mann--Whitney AUROC; None where an image has one class."""
    s=np.asarray(scores,dtype=np.float64).ravel(); y=np.asarray(positive,dtype=bool).ravel()
    n1=int(y.sum()); n0=len(y)-n1
    if not n0 or not n1:return None
    order=np.argsort(s,kind='mergesort'); ranks=np.empty(len(s),float); sorted_s=s[order]
    starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1]; ends=np.r_[starts[1:],len(s)]
    ranks[order]=np.repeat((starts+ends-1)/2+1, ends-starts)
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n0*n1))

def _rank(x: np.ndarray)->np.ndarray:
    order=np.argsort(x,kind='mergesort'); out=np.empty(len(x),float); z=x[order]; starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]; ends=np.r_[starts[1:],len(x)]
    out[order]=np.repeat((starts+ends-1)/2+1, ends-starts)
    return out

def _spearman(x:list[float],y:list[float])->float|None:
    if len(x)<3:return None
    a,b=_rank(np.array(x)),_rank(np.array(y))
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])

def mask_features(mask:np.ndarray)->dict[str,float|None]:
    y,x=np.nonzero(mask); h,w=mask.shape; n=len(y)
    if not n:return {'area_fraction':0.,'border_distance_normalized':None,'elongation':None,'compactness':None}
    border=np.minimum.reduce([y,h-1-y,x,w-1-x]).mean()/float(np.hypot(h,w))
    if n<3:return {'area_fraction':n/(h*w),'border_distance_normalized':float(border),'elongation':None,'compactness':None}
    eig=np.linalg.eigvalsh(np.cov(np.stack((y,x)))); lo,hi=float(eig[0]),float(eig[1])
    return {'area_fraction':n/(h*w),'border_distance_normalized':float(border),'elongation':None if lo<=0 else float(np.sqrt(hi/lo)),'compactness':float(n/(np.pi*(hi+1)))}

def _raw(root:Path,prefix:str,index:int,digest:str)->np.ndarray:
    p=root/f'{prefix}{index:03d}-{digest}.bin'; raw=p.read_bytes()
    if len(raw)!=SHAPE[0]*SHAPE[1]*4 or _hash(raw)!=digest:raise ValueError('raw map hash/shape mismatch')
    a=np.frombuffer(raw,dtype='<f4').reshape(SHAPE)
    if not np.isfinite(a).all():raise ValueError('nonfinite raw map')
    return a

def _manifest_rows(root:Path, p:Path, status:str, prefix:str)->list[dict[str,Any]]:
    if p.resolve().parent!=root.resolve():raise ValueError('manifest outside artifact root')
    d=_read(p)
    if d.get('status')!=status or not isinstance(d.get('maps'),list) or len(d['maps'])!=N:raise ValueError('frozen manifest contract mismatch')
    rows=d['maps']
    for i,r in enumerate(rows):
      if not isinstance(r,Mapping) or r.get('shape')!=list(SHAPE) or r.get('dtype')!='<f4' or r.get('byte_order')!='<' or not isinstance(r.get('map_sha256'),str) or len(r['map_sha256'])!=64: raise ValueError('raw map provenance mismatch')
      _raw(root,prefix,i,r['map_sha256'])
    return rows

def _mask(path:Path)->np.ndarray:
    return np.asarray(Image.open(path).convert('L').resize((SHAPE[1],SHAPE[0]),Image.Resampling.NEAREST))>0

def analyze(*,artifact_root:Path,e2_manifest:Path,superadd_manifest:Path,dataset_root:Path,run_id:str)->dict[str,Any]:
    root=Path(artifact_root).resolve()
    e2=_manifest_rows(root,Path(e2_manifest),'SPLIT_E2_TEST_PUBLIC_RAW_MAPS','g002-e2-split-test-public-raw-')
    su=_manifest_rows(root,Path(superadd_manifest),'SUPERADD_POSTHOC_DERIVED_RAW_MAPS',f'superadd-vits-posthoc-raw-{run_id}-')
    from .g002_testpub_runtime import test_public_entries
    entries=test_public_entries(Path(dataset_root))
    rows=[]
    for i,(a,b,e) in enumerate(zip(e2,su,entries)):
      if a.get('image_identity')!=e['image_identity'] or b.get('id_sha256')!=_hash(e['image_identity'].encode()) or any(a.get(k)!=e[k] or b.get(k)!=e[k] for k in ('label','source_sha256','mask_sha256')): raise ValueError('paired identity/hash mismatch')
      am=_raw(root,'g002-e2-split-test-public-raw-',i,a['map_sha256']); bm=_raw(root,f'superadd-vits-posthoc-raw-{run_id}-',i,b['map_sha256'])
      m=np.zeros(SHAPE,bool) if e['mask'] is None else _mask(e['mask']); feat=mask_features(m)
      au,bu=tie_auroc(am,m),tie_auroc(bm,m)
      rows.append({'index':i,'id_sha256':b['id_sha256'],'label':e['label'],'source_sha256':e['source_sha256'],'mask_sha256':e['mask_sha256'],'e2_map_sha256':a['map_sha256'],'superadd_map_sha256':b['map_sha256'],'e2_pixel_auroc':au,'superadd_pixel_auroc':bu,'pixel_auroc_delta':None if au is None else bu-au,**feat,'e2_map_max':float(am.max()),'superadd_map_max':float(bm.max())})
    # map maxima are converted to within-model ranks before pairing; raw score scales are not compared.
    er=_rank(np.array([r['e2_map_max'] for r in rows]))/N; sr=_rank(np.array([r['superadd_map_max'] for r in rows]))/N
    for r,x,y in zip(rows,er,sr): r['score_rank_delta']=float(y-x); r.pop('e2_map_max');r.pop('superadd_map_max')
    bad=[r for r in rows if r['pixel_auroc_delta'] is not None]
    corr={k:_spearman([r[k] for r in bad if r[k] is not None],[r['pixel_auroc_delta'] for r in bad if r[k] is not None]) for k in ('area_fraction','border_distance_normalized','elongation','compactness')}
    ranked=sorted(bad,key=lambda r:r['pixel_auroc_delta']); reps=[]
    if ranked: reps=[('e2_stronger',ranked[0]),('similar',min(ranked,key=lambda r:abs(r['pixel_auroc_delta']))),('superadd_stronger',ranked[-1])]
    record={'status':'PAIRED_RAWMAP_DESCRIPTIVE_ANALYSIS','run_id':run_id,'protocol':'FROZEN_RAW_MAPS_AND_GT_MASKS_ONLY_NO_INFERENCE_RETRAINING_OR_THRESHOLD_TUNING','privacy':'anonymized_ids_and_hashes_only_no_source_images_or_paths','shape':list(SHAPE),'counts':{'total':N,'good':24,'bad':90},'inputs':{'e2_manifest_sha256':_hash(Path(e2_manifest).read_bytes()),'superadd_manifest_sha256':_hash(Path(superadd_manifest).read_bytes())},'descriptive_only':True,'scale_robust_measure':'per-image exact tie-aware pixel AUROC and within-model image-score ranks','spearman_pixel_auroc_delta_vs_mask_features':corr,'representatives':[{'category':c,'id_sha256':r['id_sha256'],'pixel_auroc_delta':r['pixel_auroc_delta']} for c,r in reps],'interpretation':'Measured associations only; architecture causality is not established. Quantile strata and paired deltas are descriptive, not selection or tuning.'}
    return {'record':record,'rows':rows,'representatives':reps}

def _outputs(root:Path,run_id:str,result:dict[str,Any])->dict[str,Path]:
    rec=result['record']; rows=result['rows']; js=_canon(rec); digest=_hash(js); base=f'paired-rawmap-analysis-{run_id}-{digest}'
    csv_rows=[{k:('' if v is None else v) for k,v in r.items()} for r in rows]; import io
    out=io.StringIO(); w=csv.DictWriter(out,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows); cb=out.getvalue().encode()
    # raw-map-only compact heatmap panel
    panel=Image.new('RGB',(720,max(1,len(result['representatives']))*150),(255,255,255)); d=ImageDraw.Draw(panel)
    for j,(cat,row) in enumerate(result['representatives']):
        y=j*150; d.text((8,y+5),f'{cat}: {row["id_sha256"][:12]}',fill=(0,0,0))
        for x, prefix, digest, label in ((8,'g002-e2-split-test-public-raw-',row['e2_map_sha256'],'E2'),(365,f'superadd-vits-posthoc-raw-{run_id}-',row['superadd_map_sha256'],'SuperADD')):
            a=_raw(root,prefix,row['index'],digest); lo,hi=float(a.min()),float(a.max()); z=((a-lo)/(hi-lo+1e-12)*255).astype('uint8')
            im=Image.fromarray(z).resize((340,85),Image.Resampling.BILINEAR).convert('RGB'); panel.paste(im,(x,y+35)); d.text((x,y+122),label,fill=(0,0,0))
    png=__import__('io').BytesIO();panel.save(png,format='PNG'); pb=png.getvalue()
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="220"><rect width="100%" height="100%" fill="white"/><text x="30" y="45" font-size="24">Frozen paired raw-map analysis (descriptive)</text><text x="30" y="90" font-size="16">{N} same TESTpub entries · 528×2112 · tie-aware per-image pixel AUROC</text><text x="30" y="125" font-size="14">No inference, retraining, threshold tuning, source images, or paths.</text><text x="30" y="165" font-size="14">E2/SuperADD score magnitudes are only compared as within-model ranks.</text></svg>').encode()
    paths={'json':root/f'{base}.json','csv':root/f'paired-rawmap-analysis-rows-{run_id}-{_hash(cb)}.csv','svg':root/f'paired-rawmap-analysis-figure-{run_id}-{_hash(svg)}.svg','png':root/f'paired-rawmap-analysis-panel-{run_id}-{_hash(pb)}.png'}
    for p,b in zip(paths.values(),(js,cb,svg,pb)):
        if p.exists() and p.read_bytes()!=b: raise ValueError('immutable analysis artifact collision')
        if not p.exists(): p.write_bytes(b)
    return paths

def main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--e2-manifest',type=Path,required=True);p.add_argument('--superadd-manifest',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args(argv)
 r=analyze(artifact_root=a.artifact_root,e2_manifest=a.e2_manifest,superadd_manifest=a.superadd_manifest,dataset_root=a.dataset_root,run_id=a.run_id); paths=_outputs(a.artifact_root.resolve(),a.run_id,r); print(json.dumps({'status':'READY','analysis_sha256':_hash(paths['json'].read_bytes()),'representatives':r['record']['representatives']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
