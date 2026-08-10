"""Frozen, privacy-safe paired analysis of E2-Split and SuperADD raw maps.

This command only reads already persisted TESTpub maps plus masks; it never loads a
model, decodes source images, tunes a threshold, or changes either candidate.
"""
from __future__ import annotations
import argparse, csv, json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Callable
from .storage import Allocation, READY, atomic_write, preflight

import numpy as np
from PIL import Image, ImageDraw

SHAPE=(528,2112); N=114

def _canon(x: Any)->bytes: return json.dumps(x, sort_keys=True, separators=(',',':'), allow_nan=False).encode()
def _hash(b: bytes)->str: return sha256(b).hexdigest()
def _read(p: Path)->dict[str,Any]:
    try: return json.loads(p.read_text())
    except Exception as e: raise ValueError('invalid manifest') from e


E2_LEGACY_MANIFEST_SHA256="62901bcbadcfdce167b8d646e22e2f993b120eb6a20378f84a70afa822e1e03c"
def _canonical_artifact(path:Path, root:Path, *, run_id:str|None=None)->tuple[dict[str,Any],str]:
    if path.resolve().parent!=root.resolve(): raise ValueError('artifact outside root')
    raw=path.read_bytes(); d=_read(path); digest=_hash(raw)
    if raw!=_canon(d) or (run_id is not None and d.get('run_id')!=run_id): raise ValueError('noncanonical or wrong-run artifact')
    if digest not in path.name: raise ValueError('artifact filename hash mismatch')
    return d,digest

def _bindings(root:Path,e2_path:Path,su_path:Path,run_id:str)->dict[str,Any]:
    # E2 pre-dates hash-named filenames: admit this one canonical, frozen record only.
    eraw=e2_path.read_bytes(); e2=_read(e2_path)
    if _hash(eraw)!=E2_LEGACY_MANIFEST_SHA256 or eraw!=_canon(e2) or e2.get('status')!='SPLIT_E2_TEST_PUBLIC_RAW_MAPS' or e2.get('decision_id')!='DEC-SPLIT-003' or not isinstance(e2.get('freeze_sha256'),str) or len(e2['freeze_sha256'])!=64: raise ValueError('unadmitted E2 legacy evidence')
    su,su_sha=_canonical_artifact(su_path,root,run_id=run_id)
    if su.get('status')!='SUPERADD_POSTHOC_DERIVED_RAW_MAPS' or not isinstance(su.get('lineage'),Mapping): raise ValueError('SuperADD derived manifest schema')
    orig=root/f"superadd-vits-test-public-raw-{run_id}-{su.get('original_manifest_sha256')}.json"; latch=root/f"superadd-vits-test-public-latch-{run_id}-{su.get('latch_sha256')}.json"
    original,orig_sha=_canonical_artifact(orig,root,run_id=run_id); latched,latch_sha=_canonical_artifact(latch,root,run_id=run_id)
    lineage=su['lineage']
    if original.get('status')!='SUPERADD_TEST_PUBLIC_RAW_MAPS' or original.get('lineage')!=lineage or latched.get('status')!='TEST_PUBLIC_INITIAL_ATTEMPT_LATCH': raise ValueError('SuperADD referenced lineage mismatch')
    for k in ('preflight_sha256','recipe_sha256','weight_sha256','train_bank_sha256','validation_parity_sha256','evaluator_sha256','runner_source_sha256','code_git_commit'):
        if k in lineage and latched.get(k)!=lineage.get(k): raise ValueError('SuperADD latch binding mismatch')
    return {'e2_legacy_manifest_sha256':E2_LEGACY_MANIFEST_SHA256,'e2_freeze_sha256':e2['freeze_sha256'],'e2_decision_id':e2['decision_id'],'superadd_derived_manifest_sha256':su_sha,'superadd_original_manifest_sha256':orig_sha,'superadd_latch_sha256':latch_sha,'superadd_lineage':dict(lineage)}

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

def _perimeter(mask:np.ndarray)->int:
    """4-connected digital perimeter: exposed north/south/east/west pixel edges."""
    m=np.asarray(mask,bool); return int(4*m.sum()-2*(np.logical_and(m[:,1:],m[:,:-1]).sum()+np.logical_and(m[1:,:],m[:-1,:]).sum()))

def mask_features(mask:np.ndarray)->dict[str,float|None]:
    y,x=np.nonzero(mask); h,w=mask.shape; n=len(y)
    if not n:return {'area_fraction':0.,'border_distance_normalized':None,'elongation':None,'compactness':None}
    border=np.minimum.reduce([y,h-1-y,x,w-1-x]).mean()/float(np.hypot(h,w))
    perimeter=_perimeter(mask); compact=float(4*np.pi*n/(perimeter*perimeter)) if perimeter else None
    if n<3:return {'area_fraction':n/(h*w),'border_distance_normalized':float(border),'elongation':None,'compactness':compact}
    eig=np.linalg.eigvalsh(np.cov(np.stack((y,x)))); lo,hi=float(eig[0]),float(eig[1])
    return {'area_fraction':n/(h*w),'border_distance_normalized':float(border),'elongation':None if lo<=0 else float(np.sqrt(hi/lo)),'compactness':compact}

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
    bindings=_bindings(root,Path(e2_manifest),Path(superadd_manifest),run_id)
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
      rows.append({'index':i,'id_sha256':b['id_sha256'],'label':e['label'],'source_sha256':e['source_sha256'],'mask_sha256':e['mask_sha256'],'e2_map_sha256':a['map_sha256'],'superadd_map_sha256':b['map_sha256'],'e2_pixel_auroc':au,'superadd_pixel_auroc':bu,'pixel_auroc_delta':None if au is None else bu-au,**feat,'e2_map_max':float(am.max()),'superadd_map_max':float(bm.max()),'_mask':m})
    # map maxima are converted to within-model ranks before pairing; raw score scales are not compared.
    er=_rank(np.array([r['e2_map_max'] for r in rows]))/N; sr=_rank(np.array([r['superadd_map_max'] for r in rows]))/N
    for r,x,y in zip(rows,er,sr): r['score_rank_delta']=float(y-x); r.pop('e2_map_max');r.pop('superadd_map_max')
    bad=[r for r in rows if r['pixel_auroc_delta'] is not None]
    features=('area_fraction','border_distance_normalized','elongation','compactness')
    corr={k:_spearman([r[k] for r in bad if r[k] is not None],[r['pixel_auroc_delta'] for r in bad if r[k] is not None]) for k in features}
    strata={}
    for key in features:
        values=np.array([r[key] for r in bad if r[key] is not None],dtype=float)
        if len(values)<3: strata[key]={'cutpoints':[],'strata':[]}; continue
        cuts=np.quantile(values,[1/3,2/3]).tolist(); groups=[[],[],[]]
        for r in bad:
            if r[key] is not None: groups[0 if r[key]<=cuts[0] else 1 if r[key]<=cuts[1] else 2].append(r['pixel_auroc_delta'])
        strata[key]={'cutpoints':cuts,'strata':[{'name':n,'count':len(g),'mean_paired_pixel_auroc_delta':None if not g else float(np.mean(g))} for n,g in zip(('low','middle','high'),groups)]}
    ranked=sorted(bad,key=lambda r:r['pixel_auroc_delta']); reps=[]
    if ranked: reps=[('e2_stronger',ranked[0]),('similar',min(ranked,key=lambda r:abs(r['pixel_auroc_delta']))),('superadd_stronger',ranked[-1])]
    record={'status':'PAIRED_RAWMAP_DESCRIPTIVE_ANALYSIS','run_id':run_id,'protocol':'FROZEN_RAW_MAPS_AND_GT_MASKS_ONLY_NO_INFERENCE_RETRAINING_OR_THRESHOLD_TUNING','privacy':'anonymized_ids_and_hashes_only_no_source_images_or_paths','shape':list(SHAPE),'counts':{'total':N,'good':24,'bad':90},'bindings':bindings,'descriptive_only':True,'scale_robust_measure':'per-image exact tie-aware pixel AUROC and within-model image-score ranks','spearman_pixel_auroc_delta_vs_mask_features':corr,'dataset_relative_quantile_strata':strata,'compactness_definition':'4-connected digital perimeter; compactness=4*pi*area/perimeter^2','representatives':[{'category':c,'id_sha256':r['id_sha256'],'pixel_auroc_delta':r['pixel_auroc_delta']} for c,r in reps],'interpretation':'Measured associations only; architecture causality is not established. Quantile strata and paired deltas are descriptive, not selection or tuning.'}
    return {'record':record,'rows':rows,'representatives':reps}

def _outputs(root:Path,run_id:str,result:dict[str,Any],*,admit:Callable[...,Any]=preflight,writer:Callable[...,Any]=atomic_write)->dict[str,Path]:
    rec=result['record']; rows=result['rows']; js=_canon(rec); digest=_hash(js); base=f'paired-rawmap-analysis-{run_id}-{digest}'
    csv_rows=[{k:('' if v is None else v) for k,v in r.items() if not k.startswith('_')} for r in rows]; import io
    out=io.StringIO(); w=csv.DictWriter(out,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows); cb=out.getvalue().encode()
    # Raw-map + GT-mask panel: no source images. 640px native layout stays legible on mobile.
    from PIL import ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    panel=Image.new('RGB',(640,max(1,len(result['representatives']))*205),(255,255,255)); d=ImageDraw.Draw(panel)
    for j,(cat,row) in enumerate(result['representatives']):
        y=j*205; d.text((8,y+5),f'{cat} · {row["id_sha256"][:12]} · ΔAUROC {row["pixel_auroc_delta"]:+.4f}',font=font,fill=(0,0,0))
        for x, prefix, digest, label in ((8,'g002-e2-split-test-public-raw-',row['e2_map_sha256'],'E2'),(220,f'superadd-vits-posthoc-raw-{run_id}-',row['superadd_map_sha256'],'SuperADD')):
            a=_raw(root,prefix,row['index'],digest); lo,hi=float(a.min()),float(a.max()); z=((a-lo)/(hi-lo+1e-12)*255).astype('uint8')
            im=Image.fromarray(z).resize((195,120),Image.Resampling.BILINEAR).convert('RGB'); panel.paste(im,(x,y+34)); d.text((x,y+158),label+' heatmap',font=small,fill=(0,0,0))
        gt=Image.fromarray((row['_mask']*255).astype('uint8')).resize((195,120),Image.Resampling.NEAREST).convert('RGB'); panel.paste(gt,(432,y+34)); d.text((432,y+158),'GT mask',font=small,fill=(0,0,0)); d.text((8,y+180),'Independent min-max per image/pipeline: white=low, black=high.',font=font,fill=(0,0,0))
    png=__import__('io').BytesIO();panel.save(png,format='PNG'); pb=png.getvalue()
    st=rec['dataset_relative_quantile_strata']['area_fraction']['strata']; vals=[x['mean_paired_pixel_auroc_delta'] or 0 for x in st]; bars=''.join(f'<rect x="{120+i*160}" y="{205-max(0,v)*400:.1f}" width="80" height="{abs(v)*400:.1f}" fill="{'#2878b5' if v>=0 else '#c94c4c'}"/><text x="{105+i*160}" y="235" font-size="18">{st[i]['name']} n={st[i]['count']}</text><text x="{120+i*160}" y="260" font-size="18">{v:+.3f}</text>' for i,v in enumerate(vals)); svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320"><rect width="100%" height="100%" fill="white"/><text x="20" y="36" font-size="24">Paired pixel-AUROC delta</text><text x="20" y="62" font-size="18">GT area-fraction strata · descriptive only</text><text x="20" y="88" font-size="18">SuperADD − E2 · same frozen TESTpub114</text><line x1="55" y1="205" x2="610" y2="205" stroke="#222"/>{bars}<text x="20" y="282" font-size="18">No inference, tuning, source images, or selection claim.</text><text x="20" y="307" font-size="18">4-connected mask strata; blue positive, red negative.</text></svg>').encode()
    paths={'json':root/f'{base}.json','csv':root/f'paired-rawmap-analysis-rows-{run_id}-{_hash(cb)}.csv','svg':root/f'paired-rawmap-analysis-figure-{run_id}-{_hash(svg)}.svg','png':root/f'paired-rawmap-analysis-panel-{run_id}-{_hash(pb)}.png'}
    blobs=tuple((p,b) for p,b in zip(paths.values(),(js,cb,svg,pb)) if not p.exists())
    if blobs:
        total=sum(len(b) for _,b in blobs); pending=max(len(b) for _,b in blobs); source=f'exact paired evidence bytes={total}'
        proof=admit(run_id=run_id,allocations=[Allocation('artifact',total,'persistent',source,'paired-rawmap-analysis'),Allocation('artifact',pending,'transient',source,'paired-rawmap-analysis-incoming')],reserve_bytes=pending,reserve_evidence={'max_pending_atomic_write_bytes':pending,'measured_high_water_bytes':0,'runtime_or_source_citation':source})
        if Path(proof.roots['artifact']).resolve()!=root: raise ValueError('proof root changed')
        for p,b in blobs:
            if writer(p,b,proof=proof,run_id=run_id,overwrite=False).get('status')!=READY or p.read_bytes()!=b: raise ValueError('analysis atomic write failed')
    for p,b in zip(paths.values(),(js,cb,svg,pb)):
        if p.read_bytes()!=b: raise ValueError('immutable analysis artifact collision')
    return paths

def main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--e2-manifest',type=Path,required=True);p.add_argument('--superadd-manifest',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args(argv)
 r=analyze(artifact_root=a.artifact_root,e2_manifest=a.e2_manifest,superadd_manifest=a.superadd_manifest,dataset_root=a.dataset_root,run_id=a.run_id); paths=_outputs(a.artifact_root.resolve(),a.run_id,r); print(json.dumps({'status':'READY','analysis_sha256':_hash(paths['json'].read_bytes()),'representatives':r['record']['representatives']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

def cleanup_superseded_duplicates(*, artifact_root:Path, run_id:str, admit:Callable[...,Any]=preflight, writer:Callable[...,Any]=atomic_write)->dict[str,Any]:
    """Audit then remove only legacy derivative aliases proven byte-identical."""
    root=Path(artifact_root).resolve(); legacy='24d9f48b7fe3d9e82c23ff83d19debdb45e5575aaa8001238664cbe98bd225f7'
    pairs=[
      (root/f'paired-rawmap-analysis-{run_id}-{legacy}.csv',root/f'paired-rawmap-analysis-rows-{run_id}-e947df62a3e53e160916f836a07f178e175811c0bbadb10e7da8839be5fbb701.csv'),
      (root/f'paired-rawmap-analysis-{run_id}-{legacy}.png',root/f'paired-rawmap-analysis-panel-{run_id}-012d0446be4c23d3fdb3ef94e669c6041be7a695d34fa57b0cbb82f313eb4f0b.png'),
      (root/f'paired-rawmap-analysis-{run_id}-{legacy}.svg',root/f'paired-rawmap-analysis-figure-{run_id}-f9bc03d065d1e619268f990897e666df79fc4af6d0114f9591dc935af5168922.svg'),
    ]
    audit=[]
    for old,keep in pairs:
        if not old.is_file() or not keep.is_file() or old.read_bytes()!=keep.read_bytes(): raise ValueError('superseded evidence is not an exact duplicate')
        audit.append({'old_basename':old.name,'old_sha256':_hash(old.read_bytes()),'kept_basename':keep.name,'kept_sha256':_hash(keep.read_bytes()),'reason':'SUPERSEDED_BYTE_IDENTICAL_DUPLICATE'})
    record={'status':'PAIRED_RAWMAP_SUPERSEDED_DUPLICATE_CLEANUP','run_id':run_id,'records':audit,'code_git_commit':__import__('subprocess').run(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()}
    payload=_canon(record); path=root/f'paired-rawmap-analysis-cleanup-{run_id}-{_hash(payload)}.json'; source=f'exact cleanup audit bytes={len(payload)}'
    proof=admit(run_id=run_id,allocations=[Allocation('artifact',len(payload),'persistent',source,'paired-rawmap-analysis-cleanup'),Allocation('artifact',len(payload),'transient',source,'paired-rawmap-analysis-cleanup-incoming')],reserve_bytes=len(payload),reserve_evidence={'max_pending_atomic_write_bytes':len(payload),'measured_high_water_bytes':0,'runtime_or_source_citation':source})
    if Path(proof.roots['artifact']).resolve()!=root: raise ValueError('cleanup proof root changed')
    if not path.exists() and writer(path,payload,proof=proof,run_id=run_id,overwrite=False).get('status')!=READY: raise ValueError('cleanup audit write failed')
    if path.read_bytes()!=payload: raise ValueError('cleanup audit readback failed')
    for old,keep in pairs:
        old.unlink()
        if old.exists() or _hash(keep.read_bytes())!=next(x['kept_sha256'] for x in audit if x['kept_basename']==keep.name): raise ValueError('cleanup integrity failure')
    return {**record,'audit_sha256':_hash(payload)}
