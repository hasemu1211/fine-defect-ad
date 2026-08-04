"""Validation-only geometry diagnostics; never a production minimum-size claim.

Evidence: anomalib's pinned EfficientAD PDN has receptive field 33 and stride 4.
The single local candidate below uses a 16px invalid border and therefore 32px
overlap (256px tiles, 224px stride).  The AE has global context, so no fixed
overlap proves combined-map seam equivalence.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from typing import Any, Iterable, Mapping

NO_EXTERNAL_MINIMUM_AVAILABLE = "NO_EXTERNAL_MINIMUM_AVAILABLE"
PDN_RECEPTIVE_FIELD, PDN_STRIDE = 33, 4
OFFICIAL_EVIDENCE = "https://github.com/open-edge-platform/anomalib/blob/3759687e76395c4d6d239552d3bf6d72e003da78/src/anomalib/models/image/efficient_ad/torch_model.py"
LOCAL_CANDIDATE = {"tile": (256, 256), "invalid_border": (16, 16), "stride": (224, 224), "overlap": (32, 32),
                   "evidence": OFFICIAL_EVIDENCE + " (PDN RF=33/stride=4; AE global-context limitation)"}

@dataclass(frozen=True)
class ResizePlan: source: tuple[int, int]; target: tuple[int, int]
@dataclass(frozen=True)
class TilePlan: image: tuple[int, int]; tile: tuple[int, int]; stride: tuple[int, int]; overlap: tuple[int, int]

def _canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def global_resize(source, target):
    if len(source) != 2 or len(target) != 2 or min(*source, *target) <= 0: raise ValueError("invalid resize")
    return ResizePlan(tuple(source), tuple(target))

def local_candidate() -> dict[str, Any]:
    """The only candidate, explicitly not a minimum or deployment recommendation."""
    return {**LOCAL_CANDIDATE, "claim": NO_EXTERNAL_MINIMUM_AVAILABLE, "ae_global_context_limitation": True,
            "combined_map_seam_equivalence": "NOT_PROVEN_BY_FIXED_OVERLAP"}

def bounded_tiles(image, tile, *, invalid_border):
    if len(image) != 2 or len(tile) != 2 or len(invalid_border) != 2: raise ValueError("invalid geometry")
    h, w = map(int, image); th, tw = map(int, tile); by, bx = map(int, invalid_border); sh, sw = th - 2 * by, tw - 2 * bx
    if min(h, w, th, tw, sh, sw) <= 0 or min(by, bx) < 0 or th > h or tw > w: raise ValueError("invalid tile/border")
    ys, xs = list(range(0, h - th + 1, sh)), list(range(0, w - tw + 1, sw))
    if ys[-1] != h - th: ys.append(h - th)
    if xs[-1] != w - tw: xs.append(w - tw)
    return TilePlan((h, w), (th, tw), (sh, sw), (th - sh, tw - sw)), tuple((y, x, y + th, x + tw) for y in ys for x in xs)

def stitch(shape, tiles, *, plan: TilePlan, boxes, return_coverage=False):
    """Stitch only generated boxes, masking each tile's invalid border interior."""
    h, w = map(int, shape)
    if (h, w) != plan.image or not isinstance(boxes, tuple) or not boxes: raise ValueError("tile plan/box identity")
    supplied = list(tiles)
    if len(supplied) != len(boxes): raise ValueError("tile plan/box identity")
    out, count = [[0.0] * w for _ in range(h)], [[0] * w for _ in range(h)]
    by, bx = (plan.tile[0] - plan.stride[0]) // 2, (plan.tile[1] - plan.stride[1]) // 2
    seen = set()
    for item in supplied:
        if not isinstance(item, tuple) or len(item) != 2: raise ValueError("tile plan/box identity")
        box, data = item
        if box not in boxes or box in seen or not isinstance(data, (list, tuple)) or len(data) != plan.tile[0] or any(not isinstance(row, (list, tuple)) or len(row) != plan.tile[1] for row in data): raise ValueError("tile bounds/shape")
        seen.add(box); y, x, y2, x2 = box
        top, left = (0 if y == 0 else by), (0 if x == 0 else bx)
        bottom, right = (plan.tile[0] if y2 == h else plan.tile[0] - by), (plan.tile[1] if x2 == w else plan.tile[1] - bx)
        for iy in range(top, bottom):
            for ix in range(left, right):
                value = data[iy][ix]
                if not isinstance(value, (int, float)) or not isfinite(value): raise ValueError("tile values")
                out[y + iy][x + ix] += value; count[y + iy][x + ix] += 1
    if seen != set(boxes) or any(v == 0 for row in count for v in row): raise ValueError("coverage gap")
    rebuilt = [[out[y][x] / count[y][x] for x in range(w)] for y in range(h)]
    return (rebuilt, count) if return_coverage else rebuilt

def _patterns(shape):
    h, w = shape; ramp = [[float(y * w + x) for x in range(w)] for y in range(h)]
    center = [[0.0] * w for _ in range(h)]; center[h // 2][w // 2] = 1.0
    border = [[0.0] * w for _ in range(h)]; border[0][0] = border[-1][-1] = 1.0
    seams = [[0.0] * w for _ in range(h)]
    for x in range(w): seams[h // 2][x] = 1.0
    for y in range(h): seams[y][w // 2] = 1.0
    return {"coordinate_ramp": ramp, "center_impulse": center, "border_impulse": border, "seam_crossing_lines": seams}

def _tiles(data, boxes, corrupt):
    result = []
    for index, (y, x, y2, x2) in enumerate(boxes):
        tile = [row[x:x2] for row in data[y:y2]]
        if corrupt and index == 0:
            tile = [row[:] for row in tile]
            # This location is retained by tile zero and lies in the seam valid band.
            tile[-17][len(tile[0]) // 2] += 1.0
        result.append(((y, x, y2, x2), tile))
    return result

def synthetic_diagnostics(*, corrupt=False):
    """Actually tile/stitch ramps, impulses, and seam lines; all values are measured."""
    shape = (480, 480); plan, boxes = bounded_tiles(shape, LOCAL_CANDIDATE["tile"], invalid_border=LOCAL_CANDIDATE["invalid_border"])
    results, multiplicity = {}, None
    seam_y, seam_x = plan.tile[0] - LOCAL_CANDIDATE["invalid_border"][0], plan.tile[1] - LOCAL_CANDIDATE["invalid_border"][1]
    for name, source in _patterns(shape).items():
        rebuilt, counts = stitch(shape, _tiles(source, boxes, corrupt), plan=plan, boxes=boxes, return_coverage=True)
        multiplicity = counts
        errors = [[abs(rebuilt[y][x] - source[y][x]) for x in range(shape[1])] for y in range(shape[0])]
        band = [errors[y][x] for y in range(shape[0]) for x in range(shape[1]) if abs(y-seam_y) <= 1 or abs(x-seam_x) <= 1]
        results[name] = {"max_reconstruction_error": max(map(max, errors)), "max_seam_band_error": max(band)}
    return {"candidate": local_candidate(), "tile_plan": plan, "boxes": boxes, "coverage_multiplicity": {"min": min(map(min, multiplicity)), "max": max(map(max, multiplicity))},
            "valid_border": LOCAL_CANDIDATE["invalid_border"], "overlap": plan.overlap, "patterns": results,
            "claim": NO_EXTERNAL_MINIMUM_AVAILABLE}

def empirical_border_distance_diagnostic(observations: Iterable[Mapping[str, Any]], *, approved_validation_identities: Iterable[Any]) -> dict[str, Any]:
    """Derive a validation-only border from local repeated origin evidence (never test/OOD)."""
    approved={item[0] if isinstance(item,tuple) else item.get("path") if isinstance(item,Mapping) else item for item in approved_validation_identities}
    if not approved or not all(isinstance(item,str) for item in approved): raise ValueError("approved validation identities required")
    allowed={"image_identity","pixel","origin","tile_shape","score","kind","family","case","polarity"}; groups={}
    for row in observations:
        if set(row)!=allowed: raise ValueError("test/OOD/metadata forbidden")
        identity,pixel,origin,tile,kind=row["image_identity"],row["pixel"],row["origin"],row["tile_shape"],row["kind"]
        parts=identity.split("/") if isinstance(identity,str) else []
        if (identity not in approved or parts[:2]!=["validation","good"] or len(parts)!=3 or any(part in {"",".",".."} for part in parts) or any(part.casefold() in {"test","ood","private"} for part in parts) or tuple(tile)!=(256,256) or kind not in {"normal","probe_delta"} or row["family"] not in {"impulse","seam_crossing_line"} or not isinstance(row["case"],str) or row["polarity"] not in {"black_endpoint","white_endpoint"} or not all(isinstance(v,int) for v in (*pixel,*origin)) or not isfinite(row["score"])): raise ValueError("invalid validation observation")
        local=(pixel[0]-origin[0],pixel[1]-origin[1])
        if not 0<=local[0]<256 or not 0<=local[1]<256: raise ValueError("pixel is outside tile origin")
        key=(kind,identity,tuple(pixel),row["family"],row["case"],row["polarity"]); groups.setdefault(key,{}).setdefault(tuple(origin),[]).append((min(local[0],local[1],255-local[0],255-local[1]),float(row["score"])))
    if not groups or {key[0] for key in groups}!={"normal","probe_delta"}: raise ValueError("normal and probe_delta evidence required")
    evidence=[]; kind_borders={"normal":[],"probe_delta":[]}
    for key,origins in sorted(groups.items()):
        repeats=[]
        for origin,values in origins.items():
            if len(values)!=2: raise ValueError("exactly two same-origin repeats required")
            distance=values[0][0]
            if values[1][0]!=distance: raise ValueError("same-origin local distance mismatch")
            repeats.append((distance,abs(values[1][1]-values[0][1]),origin))
        eligible=[item for item in repeats if item[0]>=LOCAL_CANDIDATE["invalid_border"][0]]
        if not eligible: continue  # outer-border anchors cannot select a zero border.
        tolerance=max(item[1] for item in eligible)
        # Start candidates at the PDN lower bound; the first observed stable distance bin
        # is the earliest claim supported by this group, rather than an outer-border zero shortcut.
        candidate=(min(item[0] for item in eligible)//PDN_STRIDE)*PDN_STRIDE
        candidate=max(LOCAL_CANDIDATE["invalid_border"][0],candidate)
        if not all(item[1]<=tolerance for item in eligible): raise ValueError("no validation-only border-distance plateau")
        kind_borders[key[0]].append(candidate);evidence.append({"kind":key[0],"image_identity":key[1],"pixel":list(key[2]),"family":key[3],"case":key[4],"polarity":key[5],"tolerance":tolerance,"candidate":candidate,"same_origin_repeats":[{"distance":d,"spread":spread,"origin":list(origin)} for d,spread,origin in repeats]})
    if not all(kind_borders.values()): raise ValueError("no eligible group-local border evidence")
    candidates={kind:max(values) for kind,values in kind_borders.items()}; border=max(LOCAL_CANDIDATE["invalid_border"][0],*candidates.values())
    if not 0<=2*border<256: raise ValueError("empirical border cannot form a 256px tile plan")
    evidence_payload={"groups":evidence,"plateau_candidates":candidates}; return {"tile":(256,256),"invalid_border":(border,border),"stride":(256-2*border,256-2*border),"overlap":(2*border,2*border),"empirical_border":border,"per_kind_border":candidates,"repeatability_evidence":evidence_payload,"repeatability_evidence_sha256":sha256(_canonical(evidence_payload)).hexdigest(),"claim":NO_EXTERNAL_MINIMUM_AVAILABLE}
