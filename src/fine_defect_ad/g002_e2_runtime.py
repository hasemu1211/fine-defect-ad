"""G002 E2 validation-only tiled raw-map collection.

This module deliberately has no threshold, comparator, metric, verdict, TESTpub,
or OOD path.  It consumes only the 19 identity-admitted validation/good images.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .geometry import NO_EXTERNAL_MINIMUM_AVAILABLE, bounded_tiles, empirical_border_distance_diagnostic
from .g002_evaluate import AdmittedCheckpoint, _hash, raw_map

TILE = 256
INITIAL_BORDER = 16
COMMAND = "g002-e2-tiled-validation-raw-maps"
TRANSFORM_IDENTITY = {"decode": "RGB", "range": "[0,1]", "normalize": False, "tile": 256, "interpolation": "bilinear"}


def _np():
    import numpy as np
    return np


def decode_rgb01(path: Path):
    """Decode the admitted original image once; no crop corpus is retained."""
    from PIL import Image
    np = _np()
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not bool(np.isfinite(rgb).all()) or float(rgb.min()) < 0 or float(rgb.max()) > 1:
        raise ValueError("RGB [0,1] decode required")
    return rgb


def _map2d(value: Any):
    np = _np()
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1: array = array[0]
    if array.ndim != 2 or array.shape != (TILE, TILE) or not bool(np.isfinite(array).all()):
        raise ValueError("tile raw map must be finite 256x256")
    return array


def _tile_tensor(rgb: Any, box: tuple[int, int, int, int], torch: Any):
    np = _np(); y, x, y2, x2 = box; tile = np.ascontiguousarray(rgb[y:y2, x:x2].transpose(2, 0, 1))
    if tile.shape != (3, TILE, TILE): raise ValueError("exact 256 RGB tile required")
    return torch.from_numpy(tile).unsqueeze(0)


# The operational wrapper imports E1's reviewed admission/lease primitives rather
# than duplicating their trust boundary.
def run_e2_evaluation(args: Any, *, runtime_factory: Callable[..., Any], lease_factory: Callable[..., Any], torch_module: Any,
                      admit: Callable[..., Any], writer: Callable[..., Mapping[str, Any]], lease_event_loader: Callable[..., Any]) -> dict[str, Any]:
    """CUDA-only E2 runner.  Every admission/runtime error returns immutable STOPPED evidence."""
    from .evidence import immutable_json, new_evidence
    from .g002_eval_runtime import _lease_proof, _lease_record, load_training_identity, safe_load_checkpoint
    from .g002_evaluate import admit_completed_checkpoint
    from .g002_pilot import G002Args
    from .pilot import PilotEvidence, host_rss_bytes
    from .storage import READY, STOPPED_INCOMPLETE
    import time
    root = Path(args.artifact_root).resolve(); started=time.monotonic(); admitted=None; persisted=None; failure=None; lease_outcome="not_acquired"; events=None; lease_entered=False
    try:
        _lease_proof(root,args,admit)
        lease_directory=Path(args.lease_directory).resolve()
        if not lease_directory.is_relative_to(root) or lease_directory == root: raise ValueError("lease_directory must be an artifact-root descendant")
        identity, identity_run_id=load_training_identity(args.training_identity,root)
        if Path(args.sidecar).resolve() != Path(args.checkpoint).resolve().with_suffix(Path(args.checkpoint).suffix+".json"): raise ValueError("sidecar must be selected checkpoint sidecar")
        admitted=admit_completed_checkpoint(args.checkpoint,root,identity,args.dataset_root,args.final_attempt,args.metrics)
        if identity_run_id != admitted.run_id: raise ValueError("identity artifact lineage does not match admitted checkpoint")
        checkpoint=safe_load_checkpoint(admitted.path,admitted.checkpoint_sha256,torch_module)
        try:
            with lease_factory(lease_directory,args.run_id,COMMAND):
                lease_entered = True
                if not bool(torch_module.cuda.is_available()): raise RuntimeError("CUDA_UNAVAILABLE")
                device=torch_module.device("cuda:0")
                g002=G002Args(args.dataset_root,args.teacher_small,args.imagenette_root,args.run_id,lease_directory)
                model, _dm, _trainer, _validator=runtime_factory(g002,PilotEvidence(args.run_id,COMMAND,70_000),started,pilot_steps=None)
                model.load_state_dict(checkpoint["state_dict"]); model.eval(); model.to(device)
                def mapper(tile):
                    tile=tile.to(device)
                    with torch_module.inference_mode(): return model.model.get_maps(tile,normalize=False)
                # One pre-generation proof reserves every admitted original's exact f32 map bytes;
                # each map is then written and forgotten before the next image is decoded.
                from PIL import Image
                from .storage import Allocation
                expected = []
                for identity, _digest in admitted.validation_identities:
                    source = (admitted.dataset_root / "sheet_metal" / identity).resolve()
                    with Image.open(source) as decoded: width, height = decoded.size
                    if min(width, height) < TILE: raise ValueError("E2 requires originals at least 256px")
                    expected.append((identity, height * width * 4))
                total, pending = sum(size for _identity, size in expected), max(size for _identity, size in expected)
                source = f"exact admitted E2 f32 map bytes={total}; maximum atomic map bytes={pending}"
                map_proof = admit(run_id=args.run_id, allocations=[Allocation("artifact", total, "persistent", source, "g002-e2-streamed-maps"), Allocation("artifact", pending, "transient", source, "g002-e2-streamed-map-incoming")], reserve_bytes=pending, reserve_evidence={"max_pending_atomic_write_bytes":pending,"measured_high_water_bytes":0,"runtime_or_source_citation":source})
                if Path(map_proof.roots["artifact"]).resolve() != root: raise ValueError("fresh E2 map proof artifact root changed")
                index = {identity: number for number, (identity, _size) in enumerate(expected)}
                def sink(row):
                    destination = root / f"g002-e2-validation-raw-b{row['border']:03d}-{index[row['image_identity']]:02d}-{row['map_sha256']}.bin"
                    outcome = writer(destination, row["_bytes"], proof=map_proof, run_id=args.run_id, overwrite=False)
                    if outcome.get("status") != READY or _hash(destination) != row["map_sha256"]: raise ValueError("E2 streamed map write failed")
                    return {"path": str(destination)}
                def initial_failure_sink(value):
                    payload = _canonical(value); failure_source = f"exact E2 initial-border evidence bytes={len(payload)}"
                    failure_proof = admit(run_id=args.run_id, allocations=[Allocation("artifact",len(payload),"persistent",failure_source,"g002-e2-initial-border-evidence"),Allocation("artifact",len(payload),"transient",failure_source,"g002-e2-initial-border-evidence-incoming")], reserve_bytes=len(payload), reserve_evidence={"max_pending_atomic_write_bytes":len(payload),"measured_high_water_bytes":0,"runtime_or_source_citation":failure_source})
                    path=root/f"g002-e2-initial-border-evidence-{args.run_id}-{sha256(payload).hexdigest()}.json"; outcome=writer(path,payload,proof=failure_proof,run_id=args.run_id,overwrite=False)
                    if outcome.get("status") != READY or path.read_bytes()!=payload: raise ValueError("initial border evidence write failed")
                def phase(name, operation):
                    cuda=torch_module.cuda
                    synchronize=getattr(cuda,"synchronize",None); reset=getattr(cuda,"reset_peak_memory_stats",None)
                    if synchronize is not None: synchronize()
                    if reset is not None: reset()
                    began=time.monotonic(); value=operation()
                    if synchronize is not None: synchronize()
                    return value,{"scope":name,"synchronized_before":synchronize is not None,"synchronized_after":synchronize is not None,"peak_reset":reset is not None,"seconds":time.monotonic()-began,"allocated_bytes":int(cuda.max_memory_allocated()),"reserved_bytes":int(cuda.max_memory_reserved())}
                collected,e2_phase=phase("E2_FULL_NORMAL_PROBE_REPEAT_SEAM",lambda:collect_e2_with_one_revision(admitted,mapper,torch_module,map_sink=sink,initial_failure_sink=initial_failure_sink))
                e2_measured={"maps":collected["maps"],"cases":collected["probe_summary"]["cases"],"latency_seconds":e2_phase["seconds"],"vram":{"allocated_bytes":e2_phase["allocated_bytes"],"reserved_bytes":e2_phase["reserved_bytes"]},"phase":e2_phase,"revision":collected["revision"],"geometry":collected["geometry"],
                             "probe_recipe_sha256":sha256(_canonical([image["recipe"]["recipe_sha256"] for image in collected["probe_evidence"]])).hexdigest()}
                comparison_border=_collected_border(collected)
                e1_measured,e1_phase=phase("E1_FULL_NORMAL_PROBE_REPEAT",lambda:collect_e1_comparison(admitted,mapper,torch_module,border=comparison_border)); e1_measured["vram"]={"allocated_bytes":e1_phase["allocated_bytes"],"reserved_bytes":e1_phase["reserved_bytes"]};e1_measured["phase"]=e1_phase
                import subprocess
                git_root=Path(__file__).resolve().parents[2]; git_commit=subprocess.run(["git","-C",str(git_root),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip(); git_dirty=subprocess.run(["git","-C",str(git_root),"diff","--quiet"],check=False).returncode != 0
                hardware={"cuda_device":getattr(torch_module.cuda,"get_device_name",lambda _:"UNAVAILABLE")(0),"torch_version":str(getattr(torch_module,"__version__","UNKNOWN")),"cuda_version":str(getattr(getattr(torch_module,"version",None),"cuda","UNKNOWN")),"python":__import__("sys").version,"git_commit":git_commit,"git_dirty":git_dirty,"e2_phase_vram":e2_measured["vram"],"e1_phase_vram":e1_measured["vram"]}
                frozen=freeze_pretest_selection(e1=e1_measured,e2=e2_measured,admitted=admitted,hardware=hardware)
                freeze_payload=_canonical(frozen); freeze_source=f"exact E2 pre-test freeze bytes={len(freeze_payload)}"
                freeze_proof=admit(run_id=args.run_id,allocations=[Allocation("artifact",len(freeze_payload),"persistent",freeze_source,"g002-e2-pretest-freeze"),Allocation("artifact",len(freeze_payload),"transient",freeze_source,"g002-e2-pretest-freeze-incoming")],reserve_bytes=len(freeze_payload),reserve_evidence={"max_pending_atomic_write_bytes":len(freeze_payload),"measured_high_water_bytes":0,"runtime_or_source_citation":freeze_source})
                freeze_path=root/f"g002-e2-pretest-freeze-{args.run_id}-{frozen['freeze_sha256']}.json"; outcome=writer(freeze_path,freeze_payload,proof=freeze_proof,run_id=args.run_id,overwrite=False)
                if outcome.get("status") != READY or freeze_path.read_bytes()!=freeze_payload: raise ValueError("E2 pre-test freeze write failed")
                manifest_payload = _canonical({"status":"E2_RAW_MAPS_ONLY","run_id":args.run_id,"checkpoint":collected["checkpoint"],"maps":collected["maps"],"geometry":collected["geometry"],"probe_summary":collected["probe_summary"],"claim":NO_EXTERNAL_MINIMUM_AVAILABLE})
                # Final manifest/evidence receives its own exact proof before its first write.
                manifest_source=f"exact E2 manifest bytes={len(manifest_payload)}"
                manifest_proof=admit(run_id=args.run_id,allocations=[Allocation("artifact",len(manifest_payload),"persistent",manifest_source,"g002-e2-streamed-manifest"),Allocation("artifact",len(manifest_payload),"transient",manifest_source,"g002-e2-streamed-manifest-incoming")],reserve_bytes=len(manifest_payload),reserve_evidence={"max_pending_atomic_write_bytes":len(manifest_payload),"measured_high_water_bytes":0,"runtime_or_source_citation":manifest_source})
                manifest=root/f"g002-e2-validation-raw-maps-{args.run_id}.json"; outcome=writer(manifest,manifest_payload,proof=manifest_proof,run_id=args.run_id,overwrite=False)
                if outcome.get("status") != READY or manifest.read_bytes()!=manifest_payload: raise ValueError("E2 manifest write failed")
                persisted={"status":"E2_RAW_MAPS_ONLY","manifest":str(manifest),"map_paths":[row["artifact"] for row in collected["maps"]],"probe_summary":collected["probe_summary"],"geometry":collected["geometry"],"pretest_freeze":str(freeze_path),"selection":frozen["selection"]}
            lease_outcome="normal"
        except Exception as exc:
            failure,lease_outcome=f"RUNNER:{type(exc).__name__}:{exc}","exception"
        if lease_entered:
            try: events=_lease_record(lease_event_loader(lease_directory,args.run_id),args.run_id,lease_outcome,expected_command=COMMAND,expected_pid=os.getpid())
            except Exception as exc:
                lease_failure=f"LEASE_EVIDENCE:{type(exc).__name__}:{exc}"
                failure = lease_failure if failure is None else f"{failure}; {lease_failure}"
                lease_outcome="invalid"
    except Exception as exc: failure=f"ADMISSION:{type(exc).__name__}:{exc}"
    record=new_evidence(args.run_id,COMMAND,READY if failure is None else STOPPED_INCOMPLETE,[] if failure is None else [failure])
    record.update({"timing_seconds":time.monotonic()-started,"rss_bytes":host_rss_bytes(),"lease_outcome":lease_outcome,"claim":NO_EXTERNAL_MINIMUM_AVAILABLE,
                   "limitations":[*([] if failure is None else [failure]),"NO_EXTERNAL_MINIMUM_AVAILABLE","threshold/comparator/verdict/F1/AU-PRO/TESTpub/OOD prohibited"]})
    try: record["vram"]={"allocated_bytes":int(torch_module.cuda.max_memory_allocated()),"reserved_bytes":int(torch_module.cuda.max_memory_reserved())}
    except Exception: record["vram"]={"allocated_bytes":None,"reserved_bytes":None}
    if admitted: record.update({"checkpoint_sha256":admitted.checkpoint_sha256,"identity_sha256":admitted.identity_sha256})
    if persisted: record["raw_maps"]=persisted
    if events: record["lease_events"]=events
    # E2 final evidence is persisted in its own namespace.
    payload,digest=immutable_json(record); source=f"exact immutable E2 evaluation evidence bytes={len(payload)} sha256={digest}"
    try:
        proof=admit(run_id=args.run_id,allocations=[__import__('fine_defect_ad.storage',fromlist=['Allocation']).Allocation("artifact",len(payload),"persistent",source,"g002-e2-final-evidence"),__import__('fine_defect_ad.storage',fromlist=['Allocation']).Allocation("artifact",len(payload),"transient",source,"g002-e2-final-evidence-incoming")],reserve_bytes=len(payload),reserve_evidence={"max_pending_atomic_write_bytes":len(payload),"measured_high_water_bytes":0,"runtime_or_source_citation":source})
        target=root/f"g002-e2-validation-raw-evidence-{args.run_id}-{digest}.json"; result=writer(target,payload,proof=proof,run_id=args.run_id,overwrite=False)
        if result.get("status") != READY or target.read_bytes()!=payload: raise ValueError("immutable final evidence write failed")
        return {**record,"artifact":str(target),"artifact_sha256":digest}
    except Exception as exc: return {**record,"status":STOPPED_INCOMPLETE,"limitations":[*record["limitations"],f"EVIDENCE:{type(exc).__name__}"]}

# ---- Strict E2 evidence model (replaces the provisional convenience helpers above). ----
def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _tile_map(rgb: Any, box: tuple[int, int, int, int], mapper: Callable[[Any], tuple[Any, Any]], torch: Any):
    return _map2d(raw_map(*mapper(_tile_tensor(rgb, box, torch))))


def stream_tiled_map(rgb: Any, mapper: Callable[[Any], tuple[Any, Any]], torch: Any, *, border: int = INITIAL_BORDER):
    """One-tile-at-a-time f32 stitch; seam validity is evaluated separately per origin."""
    np = _np(); h,w=map(int,rgb.shape[:2]); plan,boxes=bounded_tiles((h,w),(TILE,TILE),invalid_border=(border,border))
    sums=np.zeros((h,w),dtype=np.float64); coverage=np.zeros((h,w),dtype=np.uint16)
    for y,x,y2,x2 in boxes:
        score=_tile_map(rgb,(y,x,y2,x2),mapper,torch); top,left=(0 if y==0 else border),(0 if x==0 else border); bottom,right=(TILE if y2==h else TILE-border),(TILE if x2==w else TILE-border)
        sums[y+top:y+bottom,x+left:x+right]+=score[top:bottom,left:right];coverage[y+top:y+bottom,x+left:x+right]+=1
    if not bool((coverage>0).all()): raise ValueError("tiled coverage gap")
    return (sums/coverage).astype("<f4",copy=False),plan,boxes,int(coverage.min()),int(coverage.max()),0.0

def _derived_probe_recipe(shape: tuple[int, int], boxes: tuple[tuple[int, int, int, int], ...], source_sha256: str, border: int = INITIAL_BORDER) -> dict[str, Any]:
    """The complete fixed probe recipe; anchors are geometry-derived, never hand-picked later."""
    h, w = shape; seams_y = sorted({y2 - border for _y, _x, y2, _x2 in boxes if y2 < h}); seams_x = sorted({x2 - border for _y, _x, _y2, x2 in boxes if x2 < w})
    seam_y, seam_x = (seams_y[0] if seams_y else h // 2), (seams_x[0] if seams_x else w // 2)
    cases = [
        {"family": "impulse", "name": "center_impulse", "point": [h // 2, w // 2]},
        {"family": "impulse", "name": "outer_border_impulse", "point": [0, 0]},
        {"family": "seam_crossing_line", "name": "horizontal_seam_line", "point": [seam_y, w // 2], "line": ["row", seam_y]},
        {"family": "seam_crossing_line", "name": "vertical_seam_line", "point": [h // 2, seam_x], "line": ["column", seam_x]},
    ]
    recipe = {"source_sha256": source_sha256, "shape": list(shape), "tile": TILE, "cases": cases,
              "polarities": ["black_endpoint", "white_endpoint"], "derivation": "initial tiled geometry boxes"}
    return {**recipe, "recipe_sha256": sha256(_canonical(recipe)).hexdigest()}


def _apply_probe(rgb: Any, case: Mapping[str, Any], polarity: str):
    np = _np(); result = rgb.copy(); value = 0.0 if polarity == "black_endpoint" else 1.0
    y, x = map(int, case["point"])
    if "line" not in case: result[y, x, :] = value
    elif case["line"][0] == "row": result[y, :, :] = value
    elif case["line"][0] == "column": result[:, x, :] = value
    else: raise ValueError("unknown preregistered probe family")
    return result


def _origins_for(point: tuple[int, int], boxes: Iterable[tuple[int, int, int, int]]) -> tuple[tuple[int, int, int, int], ...]:
    y, x = point; result = tuple(box for box in boxes if box[0] <= y < box[2] and box[1] <= x < box[3])
    if not result: raise ValueError("derived probe anchor lacks an exact tile origin")
    return result


def _interval(values: Iterable[float]) -> list[float]:
    data = [float(value) for value in values]
    if not data or not all(__import__("math").isfinite(value) for value in data): raise ValueError("finite evidence interval required")
    return [min(data), max(data)]


def per_origin_probe_evidence(rgb: Any, mapper: Callable[[Any], tuple[Any, Any]], torch: Any, *, identity: str, source_sha256: str, border: int) -> dict[str, Any]:
    """Runs every origin covering every preregistered source pixel twice per input.

    Seam values are cross-origin disagreement at that same source pixel, never
    adjacent natural-image gradients and never a box-coordinate proxy.
    """
    shape = tuple(map(int, rgb.shape[:2])); plan, boxes = bounded_tiles(shape, (TILE, TILE), invalid_border=(border, border))
    recipe = _derived_probe_recipe(shape, boxes, source_sha256, border); records = []
    for case in recipe["cases"]:
        point = tuple(case["point"]); origins = _origins_for(point, boxes)
        for polarity in recipe["polarities"]:
            probe = _apply_probe(rgb, case, polarity)
            normal_values, response_values = [], []
            group = []
            for box in origins:
                y, x, _y2, _x2 = box; ly, lx = point[0] - y, point[1] - x
                # Four actual calls per origin: normal repeat 1/2 and probe repeat 1/2.
                n1, n2 = _tile_map(rgb, box, mapper, torch), _tile_map(rgb, box, mapper, torch)
                p1, p2 = _tile_map(probe, box, mapper, torch), _tile_map(probe, box, mapper, torch)
                normals=(float(n1[ly,lx]),float(n2[ly,lx])); responses=(abs(float(p1[ly,lx]-n1[ly,lx])),abs(float(p2[ly,lx]-n2[ly,lx])))
                normal_values.extend(normals); response_values.extend(responses)
                row={"image_identity":identity,"source_sha256":source_sha256,"family":case["family"],"case":case["name"],"polarity":polarity,"pixel":list(point),"origin":[y,x],
                     "normal_repeat":list(normals),"response_repeat":list(responses),"normal_repeatability":abs(normals[1]-normals[0]),"response_repeatability":abs(responses[1]-responses[0]),
                     "response_interval":_interval(responses),"recipe_sha256":recipe["recipe_sha256"],"probe_content_sha256":sha256(_np().ascontiguousarray(probe).tobytes()).hexdigest()}
                records.append(row);group.append(row)
            # Two distinct seam measurements at the exact same source pixel, not natural gradients.
            normal_cross=max(normal_values)-min(normal_values); response_cross=max(response_values)-min(response_values)
            for row in group: row["cross_origin_normal_disagreement"]=normal_cross; row["cross_origin_response_disagreement"]=response_cross
    output = {"identity": identity, "source_sha256": source_sha256, "border": border, "recipe": recipe, "records": records}
    output["output_sha256"] = sha256(_canonical(output)).hexdigest()
    return output


def summarize_probe_cases(evidence: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pair response/repeatability intervals by identity, family, case and polarity; no global max."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for image in evidence:
        for row in image["records"]:
            key = tuple(str(row[name]) for name in ("image_identity", "family", "case", "polarity")); groups.setdefault(key, []).append(row)
    cases=[]
    for (identity,family,name,polarity), rows in sorted(groups.items()):
        cases.append({"image_identity":identity,"family":family,"case":name,"polarity":polarity,
                      "response_interval":_interval(v for row in rows for v in row["response_repeat"]),
                      "normal_repeatability":max(float(row["normal_repeatability"]) for row in rows),"response_repeatability":max(float(row["response_repeatability"]) for row in rows),
                      "cross_origin_normal_disagreement":max(float(row["cross_origin_normal_disagreement"]) for row in rows),"cross_origin_response_disagreement":max(float(row["cross_origin_response_disagreement"]) for row in rows),
                      "recipe_sha256":rows[0]["recipe_sha256"],"probe_content_sha256":rows[0]["probe_content_sha256"]})
    return {"cases": cases, "case_sha256": sha256(_canonical(cases)).hexdigest()}


def _measured_gates(value: Mapping[str, Any]) -> dict[str, bool]:
    """Derive gates exclusively from persisted-map/probe/timing measurements."""
    maps, cases = value.get("maps"), value.get("cases")
    if not isinstance(maps, list) or not isinstance(cases, list): raise ValueError("measured maps and paired cases required")
    identities = [row.get("image_identity") for row in maps if isinstance(row, Mapping)]
    integrity = len(maps) == 19 and len(set(identities)) == 19 and all(isinstance(row.get("map_sha256"), str) and len(row["map_sha256"]) == 64 for row in maps)
    coverage = integrity and all(isinstance(row.get("coverage_min"), int) and row["coverage_min"] >= 1 for row in maps)
    origin_seam = bool(cases) and all(isinstance(row.get("cross_origin_normal_disagreement"),(int,float)) and isinstance(row.get("cross_origin_response_disagreement"),(int,float)) and row["cross_origin_normal_disagreement"] <= row["normal_repeatability"] and row["cross_origin_response_disagreement"] <= row["response_repeatability"] for row in cases)
    latency, vram = value.get("latency_seconds"), value.get("vram")
    feasible = isinstance(latency, (int, float)) and latency >= 0 and isinstance(vram, Mapping) and all(isinstance(vram.get(key), int) and vram[key] >= 0 for key in ("allocated_bytes", "reserved_bytes"))
    revision = value.get("revision", {})
    if isinstance(revision, Mapping) and revision.get("e2_eligible") is False: origin_seam = False
    return {"integrity_valid":integrity,"coverage_valid":coverage,"origin_seam_valid":origin_seam,"feasible":feasible}


def select_e1_or_e2(*, e1: Mapping[str, Any], e2: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen hierarchical, non-weighted pre-TESTpub rule from measured artifacts only."""
    forbidden=("test","ood","threshold","comparator","f1","aupro","verdict","minimum")
    if any(any(token in str(key).casefold() for token in forbidden) for source in (e1,e2) for key in source): raise ValueError("selection cannot consult prohibited inputs")
    gates_one,gates_two=_measured_gates(e1),_measured_gates(e2)
    e1_by={tuple(item[key] for key in ("image_identity","family","case","polarity")):item for item in e1["cases"]}
    e2_by={tuple(item[key] for key in ("image_identity","family","case","polarity")):item for item in e2["cases"]}
    if set(e1_by) != set(e2_by) or not e1_by: raise ValueError("paired preregistered case sets required")
    non_worse=True; exceeds={"impulse":False,"seam_crossing_line":False}
    for key in sorted(e1_by):
        one,two=e1_by[key],e2_by[key]
        if one.get("recipe_sha256") != two.get("recipe_sha256") or one.get("probe_content_sha256") != two.get("probe_content_sha256"):
            raise ValueError("E1/E2 recipe or probe content mismatch")
        a,b=map(float,one["response_interval"]); c,_d=map(float,two["response_interval"])
        non_worse &= c >= a
        if c > b + max(float(one["response_repeatability"]),float(two["response_repeatability"])): exceeds[key[1]]=True
    all_gates=all(gates_one.values()) and all(gates_two.values())
    selected="E2" if all_gates and non_worse and all(exceeds.values()) else "E1"
    return {"selected":selected,"rule":"HIERARCHICAL_PAIRED_VALIDATION_ONLY_NONWEIGHTED","measured_gates":{"E1":gates_one,"E2":gates_two},"non_worse_every_case":non_worse,
            "beyond_repeatability_by_family":exceeds,"claim":NO_EXTERNAL_MINIMUM_AVAILABLE,
            "limitations":["NO_EXTERNAL_MINIMUM_AVAILABLE","minimum-defect and production claims prohibited","latency/VRAM are feasibility/reporting only"]}

def collect_e2(admitted: AdmittedCheckpoint, mapper: Callable[[Any], tuple[Any, Any]], torch: Any, *, map_sink: Callable[[Mapping[str, Any]], Mapping[str, Any]], border: int = INITIAL_BORDER) -> dict[str, Any]:
    """Map-at-a-time E2 collection.  A sink is mandatory so 19 raw maps are never retained."""
    if not isinstance(border, int) or not 0 <= 2 * border < TILE: raise ValueError("invalid E2 border")
    images=[]; metadata=[]; geometry_rows=[]
    for identity, source_sha in admitted.validation_identities:
        source=(admitted.dataset_root/"sheet_metal"/identity).resolve()
        if _hash(source)!=source_sha: raise ValueError("source content hash mismatch")
        rgb=decode_rgb01(source); mapped, plan, boxes, cmin,cmax,seam=stream_tiled_map(rgb,mapper,torch,border=border)
        evidence=per_origin_probe_evidence(rgb,mapper,torch,identity=identity,source_sha256=source_sha,border=border)
        raw=mapped.astype("<f4",copy=False).tobytes(order="C")
        row={"image_identity":identity,"source_sha256":source_sha,"map_sha256":sha256(raw).hexdigest(),"dtype":"<f4","shape":list(mapped.shape),"byte_order":"<","_bytes":raw,"coverage_min":cmin,"coverage_max":cmax,"seam_max_abs":seam,"border":border}
        written=map_sink(row); metadata.append({key:value for key,value in row.items() if key!="_bytes"}|{"artifact":written.get("path")}); images.append(evidence)
        # Geometry.py accepts strictly validation-only rows; real values are per-origin rerun outputs.
        for probe in evidence["records"]:
            for kind, values in (("normal",probe["normal_repeat"]),("probe_delta",probe["response_repeat"])):
                for score in values: geometry_rows.append({"image_identity":identity,"pixel":tuple(probe["pixel"]),"origin":tuple(probe["origin"]),"tile_shape":(TILE,TILE),"score":score,"kind":kind,"family":probe["family"],"case":probe["case"],"polarity":probe["polarity"]})
    geometry=empirical_border_distance_diagnostic(geometry_rows,approved_validation_identities=admitted.validation_identities)
    # A caller performs the one permitted full re-run if geometry requests it; this function never iterates.
    return {"status":"E2_RAW_MAPS_ONLY","maps":metadata,"probe_evidence":images,"probe_summary":summarize_probe_cases(images),"geometry":geometry,
            "checkpoint":{key:getattr(admitted,key) for key in ("checkpoint_sha256","sidecar_sha256","metrics_sha256","final_attempt_sha256","identity_sha256","pilot_sha256")},"transform_identity":TRANSFORM_IDENTITY,"claim":NO_EXTERNAL_MINIMUM_AVAILABLE}


def collect_e2_with_one_revision(admitted: AdmittedCheckpoint, mapper: Callable[[Any], tuple[Any, Any]], torch: Any, *, map_sink: Callable[[Mapping[str, Any]], Mapping[str, Any]], initial_failure_sink: Callable[[Mapping[str, Any]], None]) -> dict[str, Any]:
    """Run diagnostics without artifacts, then persist only the one final selected E2 attempt."""
    initial = collect_e2(admitted, mapper, torch, map_sink=lambda _row: {"path": None}, border=INITIAL_BORDER)
    revised = int(initial["geometry"]["empirical_border"])
    if revised == INITIAL_BORDER:
        final = collect_e2(admitted, mapper, torch, map_sink=map_sink, border=INITIAL_BORDER)
        return {**final, "revision": {"attempts": 1, "diagnostic_passes": 1, "initial_border": INITIAL_BORDER, "status": "INITIAL_CONSTRAINTS_MET", "e2_eligible": True}}
    initial_failure_sink({"status":"INITIAL_BORDER_CONSTRAINT_FAILED","border":INITIAL_BORDER,"failed_constraints":[{"name":"empirical_border","required":INITIAL_BORDER,"observed":revised},{"name":"stride","required":TILE-2*INITIAL_BORDER,"observed":initial["geometry"].get("stride", (TILE-2*revised,))[0]}],"geometry":initial["geometry"],"probe_summary":initial["probe_summary"],"claim":NO_EXTERNAL_MINIMUM_AVAILABLE})
    second = collect_e2(admitted, mapper, torch, map_sink=map_sink, border=revised)
    if int(second["geometry"]["empirical_border"]) != revised:
        return {**second, "revision": {"attempts": 2, "initial_border": INITIAL_BORDER, "revised_border": revised, "status": "REVISION_UNSTABLE_RETAIN_E1", "e2_eligible": False}}
    return {**second, "revision": {"attempts": 2, "initial_border": INITIAL_BORDER, "revised_border": revised, "status": "ONE_REVISION_COMPLETE", "e2_eligible": True}}

def _collected_border(collected: Mapping[str, Any]) -> int:
    borders = {row.get("border") for row in collected.get("maps", []) if isinstance(row, Mapping)}
    if len(borders) != 1:
        raise ValueError("E2 maps require one uniform border")
    border = borders.pop()
    if not isinstance(border, int) or isinstance(border, bool) or not 0 <= 2 * border < TILE:
        raise ValueError("E2 map border invalid")
    return border


def _resize_rgb01(rgb: Any, torch: Any):
    """Reviewed E1 path: torchvision Resize(256, bilinear, antialias=True), no PIL approximation."""
    if tuple(rgb.shape[:2]) == (TILE, TILE): return rgb
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    value=torch.from_numpy(_np().ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0)
    transformed=resize(value,[TILE,TILE],interpolation=InterpolationMode.BILINEAR,antialias=True)
    array=transformed.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    if tuple(array.shape)!=(TILE,TILE,3): raise ValueError("pinned E1 resize output identity failed")
    return array

def collect_e1_comparison(admitted: AdmittedCheckpoint, mapper: Callable[[Any], tuple[Any, Any]], torch: Any, *, border: int = INITIAL_BORDER) -> dict[str, Any]:
    """Validation-only E1 normal/probe counterpart for the frozen paired decision."""
    import time
    started=time.monotonic(); images=[]; maps=[]
    for identity, source_sha in admitted.validation_identities:
        source=(admitted.dataset_root/"sheet_metal"/identity).resolve()
        if _hash(source)!=source_sha: raise ValueError("source content hash mismatch")
        rgb=decode_rgb01(source); _plan, boxes=bounded_tiles(tuple(rgb.shape[:2]),(TILE,TILE),invalid_border=(border,border))
        recipe=_derived_probe_recipe(tuple(rgb.shape[:2]),boxes,source_sha,border); normal=_resize_rgb01(rgb,torch); base1=_tile_map(normal,(0,0,TILE,TILE),mapper,torch); base2=_tile_map(normal,(0,0,TILE,TILE),mapper,torch); records=[]
        for case in recipe["cases"]:
            sy,sx=case["point"]; y=min(TILE-1,round(sy*(TILE-1)/max(1,rgb.shape[0]-1)));x=min(TILE-1,round(sx*(TILE-1)/max(1,rgb.shape[1]-1)))
            for polarity in recipe["polarities"]:
                first=_tile_map(_resize_rgb01(_apply_probe(rgb,case,polarity),torch),(0,0,TILE,TILE),mapper,torch)
                second=_tile_map(_resize_rgb01(_apply_probe(rgb,case,polarity),torch),(0,0,TILE,TILE),mapper,torch)
                response=(abs(float(first[y,x]-base1[y,x])),abs(float(second[y,x]-base2[y,x])))
                records.append({"image_identity":identity,"source_sha256":source_sha,"family":case["family"],"case":case["name"],"polarity":polarity,"pixel":[sy,sx],"origin":[0,0],"normal_repeat":[float(base1[y,x]),float(base2[y,x])],"response_repeat":[response[0],response[1]],"normal_repeatability":abs(float(base2[y,x]-base1[y,x])),"response_repeatability":abs(response[1]-response[0]),"response_interval":_interval(response),"cross_origin_normal_disagreement":0.0,"cross_origin_response_disagreement":0.0,"recipe_sha256":recipe["recipe_sha256"],"probe_content_sha256":sha256(_np().ascontiguousarray(_apply_probe(rgb,case,polarity)).tobytes()).hexdigest()})
        raw=base1.astype("<f4",copy=False).tobytes(order="C");images.append({"records":records});maps.append({"image_identity":identity,"map_sha256":sha256(raw).hexdigest(),"coverage_min":1})
    return {"maps":maps,"cases":summarize_probe_cases(images)["cases"],"latency_seconds":time.monotonic()-started,"probe_recipe_sha256":sha256(_canonical([image["records"][0]["recipe_sha256"] for image in images])).hexdigest()}


def freeze_pretest_selection(*, e1: Mapping[str, Any], e2: Mapping[str, Any], admitted: AdmittedCheckpoint, hardware: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical pre-TEST freeze binding measured summaries, geometry, lineage and hardware identity."""
    chosen=select_e1_or_e2(e1=e1,e2=e2); e1_summary=dict(e1);e2_summary=dict(e2)
    e1_hash,e2_hash=sha256(_canonical(e1_summary)).hexdigest(),sha256(_canonical(e2_summary)).hexdigest()
    identities=[{"path":path,"sha256":digest} for path,digest in admitted.validation_identities]
    payload={"stage":"PRE_TEST_FREEZE","status":"FROZEN","decision_id":"DEC-GEO-002","selection":chosen,"checkpoint_sha256":admitted.checkpoint_sha256,"sidecar_sha256":admitted.sidecar_sha256,"metrics_sha256":admitted.metrics_sha256,"final_attempt_sha256":admitted.final_attempt_sha256,"identity_sha256":admitted.identity_sha256,"pilot_sha256":admitted.pilot_sha256,"validation_identities":identities,"hardware":dict(hardware),"e1_measurement":e1_summary,"e1_measurement_sha256":e1_hash,"e2_measurement":e2_summary,"e2_measurement_sha256":e2_hash,"geometry":e2_summary.get("geometry"),"revision":e2_summary.get("revision"),"claim":NO_EXTERNAL_MINIMUM_AVAILABLE,"limitations":["NO_EXTERNAL_MINIMUM_AVAILABLE","TESTpub/OOD/threshold/comparator/verdict prohibited before this freeze"]}
    result={**payload,"freeze_sha256":sha256(_canonical(payload)).hexdigest()}; verify_pretest_freeze(result); return result


def verify_pretest_freeze(value: Mapping[str, Any]) -> None:
    """Reject tampered measurement/geometry/lineage freeze records before READY."""
    required={"freeze_sha256","stage","status","decision_id","selection","e1_measurement","e1_measurement_sha256","e2_measurement","e2_measurement_sha256","geometry","revision","checkpoint_sha256","sidecar_sha256","metrics_sha256","final_attempt_sha256","identity_sha256","pilot_sha256","validation_identities","hardware"}
    if not required <= set(value) or value["stage"] != "PRE_TEST_FREEZE" or value["status"] != "FROZEN" or value["decision_id"] != "DEC-GEO-002": raise ValueError("incomplete pre-test freeze")
    selection=value["selection"]
    if not isinstance(selection, Mapping) or selection.get("selected") not in {"E1","E2"}: raise ValueError("invalid pre-test freeze selection")
    hashes=("freeze_sha256","checkpoint_sha256","sidecar_sha256","metrics_sha256","final_attempt_sha256","identity_sha256","pilot_sha256","e1_measurement_sha256","e2_measurement_sha256")
    if not all(isinstance(value[key],str) and value[key] for key in hashes): raise ValueError("invalid pre-test freeze hash")
    identities=value["validation_identities"]
    if not isinstance(identities,list) or len(identities) != 19 or len({row.get("path") for row in identities if isinstance(row,Mapping)}) != 19 or any(not isinstance(row,Mapping) or set(row)!={"path","sha256"} or not isinstance(row["path"],str) or not row["path"].startswith("validation/good/") or not isinstance(row["sha256"],str) or not row["sha256"] for row in identities): raise ValueError("invalid pre-test freeze identities")
    if sha256(_canonical(value["e1_measurement"])).hexdigest()!=value["e1_measurement_sha256"] or sha256(_canonical(value["e2_measurement"])).hexdigest()!=value["e2_measurement_sha256"]: raise ValueError("measurement summary hash mismatch")
    expected_selection=select_e1_or_e2(e1=value["e1_measurement"],e2=value["e2_measurement"])
    if value["selection"] != expected_selection: raise ValueError("pre-test freeze selection is not derived from measurements")
    if value["geometry"] != value["e2_measurement"].get("geometry") or value["revision"] != value["e2_measurement"].get("revision"): raise ValueError("pre-test freeze E2 geometry/revision mismatch")
    core={key:value[key] for key in value if key!="freeze_sha256"}
    if sha256(_canonical(core)).hexdigest()!=value["freeze_sha256"]: raise ValueError("freeze hash mismatch")

def parse_args(argv: Iterable[str] | None = None):
    """Production CLI: shares the reviewed E1 required-input contract."""
    from .g002_eval_runtime import parse_args as parse_e1_args
    return parse_e1_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    from .g002_pilot import _lazy_runtime
    from .gpu_lock import GpuLease
    from .storage import atomic_write, preflight, READY
    import torch
    from .pilot import lease_events
    result = run_e2_evaluation(args, runtime_factory=_lazy_runtime, lease_factory=GpuLease, torch_module=torch, admit=preflight, writer=atomic_write, lease_event_loader=lease_events)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())


# DEC-SPLIT-003 is intentionally separate from DEC-GEO-002.  The latter is
# historical E1 evidence and must remain reproducible.
SPLIT_DECISION_ID = "DEC-SPLIT-003"
SPLIT_STRIDE = 128
SPLIT_TARGET_SHAPE = (528, 2112)  # H, W: half of the studied sheet-metal size


def _split_boxes(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], ...]:
    """Return deterministic 256px terminal tiles at 128px stride."""
    h, w = map(int, shape)
    if min(h, w) < TILE:
        raise ValueError("split branch requires source dimensions >= 256")
    ys = list(range(0, h - TILE + 1, SPLIT_STRIDE))
    xs = list(range(0, w - TILE + 1, SPLIT_STRIDE))
    if ys[-1] != h - TILE:
        ys.append(h - TILE)
    if xs[-1] != w - TILE:
        xs.append(w - TILE)
    return tuple((y, x, y + TILE, x + TILE) for y in ys for x in xs)


def periodic_hann_weights(box: tuple[int, int, int, int], shape: tuple[int, int]):
    """Positive, separable periodic-Hann tile weights, including image edges."""
    np = _np()
    y, x, y2, x2 = box
    h, w = map(int, shape)
    if (y2 - y, x2 - x) != (TILE, TILE):
        raise ValueError("split tile must be exactly 256x256")
    # Half-sample periodic Hann has no zero samples.  Edge variants retain the
    # same taper on overlaps while making the physical image boundary constant.
    axis = np.sin(np.pi * (np.arange(TILE, dtype=np.float64) + 0.5) / TILE) ** 2
    wy, wx = axis.copy(), axis.copy()
    if y == 0: wy[0] = 1.0
    if y2 == h: wy[-1] = 1.0
    if x == 0: wx[0] = 1.0
    if x2 == w: wx[-1] = 1.0
    weight = wy[:, None] * wx[None, :]
    if not bool(np.isfinite(weight).all()) or float(weight.min()) <= 0:
        raise ValueError("split Hann weights must be finite and positive")
    return weight


def _core_model(model: Any) -> Any:
    """Accept either the Lightning wrapper or its EfficientAD torch module."""
    return getattr(model, "model", model)


def local_st_map(tile: Any, model: Any, torch: Any):
    """Exact EfficientAD raw ST branch only; AE is deliberately never called."""
    core = _core_model(model)
    import torch.nn.functional as F
    with torch.inference_mode():
        teacher = core.teacher(tile)
        if core.is_set(core.mean_std):
            teacher = (teacher - core.mean_std["mean"]) / core.mean_std["std"]
        student = core.student(tile)
        distance = (teacher - student[:, :core.teacher_out_channels]) ** 2
        value = distance.mean(dim=1, keepdim=True)
        if core.pad_maps:
            value = F.pad(value, (4, 4, 4, 4))
        return F.interpolate(value, size=tile.shape[-2:], mode="bilinear")


def global_stae_map(canonical_tile: Any, model: Any, torch: Any):
    """Exact EfficientAD raw STAE branch once on the canonical 256px image."""
    core = _core_model(model)
    import torch.nn.functional as F
    with torch.inference_mode():
        student = core.student(canonical_tile)
        ae = core.ae(canonical_tile, canonical_tile.shape[-2:])
        value = ((ae - student[:, core.teacher_out_channels:]) ** 2).mean(dim=1, keepdim=True)
        if core.pad_maps:
            value = F.pad(value, (4, 4, 4, 4))
        return F.interpolate(value, size=canonical_tile.shape[-2:], mode="bilinear")


def canonical_256(rgb: Any, torch: Any, *, device: Any | None = None):
    """Pinned RGB [0,1] -> 1x3x256x256 bilinear-antialias transform."""
    import torch.nn.functional as F
    np = _np()
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3 or not bool(np.isfinite(array).all()):
        raise ValueError("canonical transform requires finite HxWx3 RGB")
    value = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    if device is not None:
        value = value.to(device)
    return F.interpolate(value, size=(TILE, TILE), mode="bilinear", align_corners=False, antialias=True)


def split_branch_raw_maps(rgb: Any, model: Any, torch: Any, *, device: Any | None = None):
    """Produce stitched local ST and global canonical STAE maps without AE tiling."""
    np = _np()
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3 or not bool(np.isfinite(array).all()):
        raise ValueError("split branch requires finite HxWx3 RGB")
    h, w = array.shape[:2]
    boxes = _split_boxes((h, w))
    sums = np.zeros((h, w), dtype=np.float64)
    weights = np.zeros((h, w), dtype=np.float64)
    for box in boxes:
        tile = _tile_tensor(array, box, torch)
        if device is not None:
            tile = tile.to(device)
        value = _map2d(local_st_map(tile, model, torch))
        y, x, y2, x2 = box
        weight = periodic_hann_weights(box, (h, w))
        sums[y:y2, x:x2] += value * weight
        weights[y:y2, x:x2] += weight
    if not bool(np.isfinite(weights).all()) or float(weights.min()) <= 0:
        raise ValueError("split blend has incomplete or non-positive coverage")
    local = (sums / weights).astype("<f4", copy=False)
    global_map = _map2d(global_stae_map(canonical_256(array, torch, device=device), model, torch))
    if not bool(np.isfinite(local).all()) or not bool(np.isfinite(global_map).all()):
        raise ValueError("split branch maps must be finite")
    return local, global_map, {"tile": TILE, "stride": SPLIT_STRIDE, "boxes": [list(box) for box in boxes],
                               "blend": "separable_periodic_hann_positive_boundary_variants", "weight_min": float(weights.min()),
                               "weight_max": float(weights.max())}


def split_quantiles(local_maps: Iterable[Any], global_maps: Iterable[Any]) -> dict[str, float]:
    """Fresh q90/q99.5 calibration; checkpoint quantiles are never consulted."""
    np = _np()
    def quantile(value: Any, q: float) -> float:
        try:
            return float(np.quantile(value, q, method="linear"))
        except TypeError:
            return float(np.quantile(value, q, interpolation="linear"))
    def values(items: Iterable[Any], name: str) -> tuple[float, float]:
        data = [np.asarray(item, dtype=np.float32).ravel() for item in items]
        if not data:
            raise ValueError(f"{name} validation maps required")
        joined = np.concatenate(data).astype(np.float32, copy=False)
        if not bool(np.isfinite(joined).all()):
            raise ValueError(f"{name} validation maps must be finite")
        qa, qb = quantile(joined, .90), quantile(joined, .995)
        if not np.isfinite(qa) or not np.isfinite(qb) or not qb > qa:
            raise ValueError(f"{name} q99.5 must be greater than q90")
        return qa, qb
    qa_st, qb_st = values(local_maps, "local_st")
    qa_ae, qb_ae = values(global_maps, "global_stae")
    return {"qa_st": qa_st, "qb_st": qb_st, "qa_stae": qa_ae, "qb_stae": qb_ae}


def combine_split_maps(local: Any, global_map: Any, quantiles: Mapping[str, float], torch: Any):
    """Apply the pinned EfficientAD normalization, resize to 2112x528, average."""
    import torch.nn.functional as F
    np = _np()
    needed = ("qa_st", "qb_st", "qa_stae", "qb_stae")
    if set(needed) - set(quantiles):
        raise ValueError("split quantiles incomplete")
    if not all(np.isfinite(float(quantiles[k])) for k in needed) or not float(quantiles["qb_st"]) > float(quantiles["qa_st"]) or not float(quantiles["qb_stae"]) > float(quantiles["qa_stae"]):
        raise ValueError("split quantiles invalid")
    def resize(value: Any, qa: float, qb: float):
        array = np.asarray(value, dtype=np.float32)
        item = torch.from_numpy(np.ascontiguousarray(array)).reshape(1, 1, *array.shape)
        item = .1 * (item - qa) / (qb - qa)
        return F.interpolate(item, size=SPLIT_TARGET_SHAPE, mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy()
    st = resize(local, float(quantiles["qa_st"]), float(quantiles["qb_st"]))
    stae = resize(global_map, float(quantiles["qa_stae"]), float(quantiles["qb_stae"]))
    result = np.ascontiguousarray(.5 * (st + stae), dtype="<f4")
    if result.shape != SPLIT_TARGET_SHAPE or not bool(np.isfinite(result).all()):
        raise ValueError("split output must be finite 528x2112")
    return result


def freeze_split_validation(*, admitted: AdmittedCheckpoint, quantiles: Mapping[str, float], map_rows: Iterable[Mapping[str, Any]], geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Create the immutable, validation-only admission record for split E2.

    This is deliberately a READY gate, not a model-selection comparison: no
    TESTpub-derived value is accepted anywhere in its inputs.
    """
    rows = [dict(row) for row in map_rows]
    expected = dict(admitted.validation_identities)
    if len(rows) != 19 or {row.get("image_identity") for row in rows} != set(expected):
        raise ValueError("exactly the admitted 19 validation/good maps are required")
    for row in rows:
        if row.get("source_sha256") != expected[row["image_identity"]] or not isinstance(row.get("local_st_sha256"), str) or not isinstance(row.get("global_stae_sha256"), str):
            raise ValueError("validation source/map hash binding invalid")
    # Re-run the validation rather than trusting a serialized q value.
    checked = split_quantiles([row["_local_st"] for row in rows], [row["_global_stae"] for row in rows])
    if dict(quantiles) != checked:
        raise ValueError("validation quantiles are not the fresh raw-map quantiles")
    if not isinstance(geometry, Mapping) or geometry.get("tile") != TILE or geometry.get("stride") != SPLIT_STRIDE or float(geometry.get("weight_min", 0)) <= 0:
        raise ValueError("split geometry gate invalid")
    maps = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    payload = {"stage": "PRE_TEST_FREEZE", "status": "READY", "decision_id": SPLIT_DECISION_ID,
               "checkpoint_sha256": admitted.checkpoint_sha256, "sidecar_sha256": admitted.sidecar_sha256,
               "metrics_sha256": admitted.metrics_sha256, "final_attempt_sha256": admitted.final_attempt_sha256,
               "identity_sha256": admitted.identity_sha256, "pilot_sha256": admitted.pilot_sha256,
               "validation_identities": [{"path": p, "sha256": s} for p, s in admitted.validation_identities],
               "geometry": dict(geometry), "quantiles": checked, "maps": maps,
               "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "protocol": "SPLIT_ST_SOURCE_RESOLUTION__GLOBAL_STAE_CANONICAL_256__NO_TEST_ACCESS"}
    return {**payload, "freeze_sha256": sha256(_canonical(payload)).hexdigest()}


def verify_split_freeze(value: Mapping[str, Any]) -> None:
    required = {"stage", "status", "decision_id", "checkpoint_sha256", "validation_identities", "geometry", "quantiles", "maps", "code_sha256", "freeze_sha256"}
    if not required <= set(value) or value["stage"] != "PRE_TEST_FREEZE" or value["status"] != "READY" or value["decision_id"] != SPLIT_DECISION_ID:
        raise ValueError("split validation freeze is incomplete")
    if sha256(_canonical({k: v for k, v in value.items() if k != "freeze_sha256"})).hexdigest() != value["freeze_sha256"]:
        raise ValueError("split validation freeze hash mismatch")
    q = value["quantiles"]
    if not isinstance(q, Mapping) or not float(q.get("qb_st", 0)) > float(q.get("qa_st", 0)) or not float(q.get("qb_stae", 0)) > float(q.get("qa_stae", 0)):
        raise ValueError("split validation freeze quantiles invalid")
