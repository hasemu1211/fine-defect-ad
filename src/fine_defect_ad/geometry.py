"""Evidence-bound global-resize and tiled geometry primitives.

This module intentionally contains no inspection-scale or overlap defaults: callers
must supply the model/input and border evidence used to create E2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

NO_EXTERNAL_MINIMUM_AVAILABLE = "NO_EXTERNAL_MINIMUM_AVAILABLE"


@dataclass(frozen=True)
class BorderEvidence:
    """Pixels invalid at each model-input border, with its recorded source."""

    top: int
    bottom: int
    left: int
    right: int
    source_locator: str

    def __post_init__(self) -> None:
        if not self.source_locator or any(not isinstance(value, int) or value < 0
                                          for value in (self.top, self.bottom, self.left, self.right)):
            raise ValueError("border evidence requires non-negative integer pixels and a source locator")


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class TilePlan:
    source_shape: tuple[int, int]
    model_shape: tuple[int, int]
    border: BorderEvidence
    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class GeometryCandidate:
    identifier: str
    kind: str
    source_shape: tuple[int, int]
    model_shape: tuple[int, int]
    external_minimum_status: str
    plan: TilePlan | None = None


def _shape(name: str, value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2 or any(not isinstance(item, int) or item < 1 for item in value):
        raise ValueError(f"{name} must be a positive (height, width) pair")
    return value


def _starts(length: int, tile: int, before: int, after: int) -> tuple[int, ...]:
    if length < tile:
        raise ValueError("tiled E2 requires source dimensions at least model-input dimensions")
    stride = tile - before - after
    if stride < 1:
        raise ValueError("border evidence leaves no valid tile interior")
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def global_resize_descriptor(source_shape: tuple[int, int], model_shape: tuple[int, int]) -> GeometryCandidate:
    """Return the fixed E1 baseline; resize operation belongs to the caller/model transform."""
    return GeometryCandidate("E1", "global_resize", _shape("source_shape", source_shape),
                             _shape("model_shape", model_shape), NO_EXTERNAL_MINIMUM_AVAILABLE)


def derive_tile_plan(source_shape: tuple[int, int], model_shape: tuple[int, int], border: BorderEvidence,
                     *, overlap: object | None = None) -> TilePlan:
    """Derive E2 coordinates from evidence; caller-proposed overlap is never accepted."""
    if overlap is not None:
        raise ValueError("overlap is derived from invalid-border evidence, not a percentage or test result")
    source_shape = _shape("source_shape", source_shape)
    model_shape = _shape("model_shape", model_shape)
    ys = _starts(source_shape[0], model_shape[0], border.top, border.bottom)
    xs = _starts(source_shape[1], model_shape[1], border.left, border.right)
    height, width = model_shape
    return TilePlan(source_shape, model_shape, border,
                    tuple(Tile(y, y + height, x, x + width) for y in ys for x in xs))


def geometry_candidates(source_shape: tuple[int, int], model_shape: tuple[int, int], border: BorderEvidence,
                        *, external_minimum: object | None = None) -> tuple[GeometryCandidate, GeometryCandidate]:
    """Return E1 and exactly one evidence-derived, exploratory E2 candidate.

    A physical minimum is merely recorded by its legitimate source upstream; it cannot
    alter this data-free derivation or create a second candidate.
    """
    if external_minimum is not None:
        locator = getattr(external_minimum, "source_locator", None)
        if not isinstance(locator, str) or not locator:
            raise ValueError("external inspection scale requires a legitimate source_locator")
    baseline = global_resize_descriptor(source_shape, model_shape)
    plan = derive_tile_plan(source_shape, model_shape, border)
    return baseline, GeometryCandidate("E2", "tiled_exploratory", plan.source_shape, plan.model_shape,
                                       NO_EXTERNAL_MINIMUM_AVAILABLE if external_minimum is None else "EXTERNAL_SOURCE_DECLARED",
                                       plan)


def valid_region(tile: Tile, plan: TilePlan) -> Tile:
    """Map the evidence-valid part of one tile to source coordinates.

    Source edges retain their available pixels; internal tile borders are trimmed.
    """
    h, w = plan.source_shape
    border = plan.border
    return Tile(tile.y0 + (border.top if tile.y0 else 0),
                tile.y1 - (border.bottom if tile.y1 < h else 0),
                tile.x0 + (border.left if tile.x0 else 0),
                tile.x1 - (border.right if tile.x1 < w else 0))


def stitch_tiles(plan: TilePlan, tile_maps: Sequence[np.ndarray]) -> np.ndarray:
    """Trim internal invalid borders and deterministically average any final overlap."""
    if len(tile_maps) != len(plan.tiles):
        raise ValueError("tile map count must match plan")
    output = np.zeros(plan.source_shape, dtype=np.float64)
    count = np.zeros(plan.source_shape, dtype=np.uint32)
    expected = plan.model_shape
    for tile, values in zip(plan.tiles, tile_maps, strict=True):
        values = np.asarray(values)
        if values.shape != expected:
            raise ValueError(f"tile map shape must be {expected}, got {values.shape}")
        region = valid_region(tile, plan)
        if region.y0 >= region.y1 or region.x0 >= region.x1:
            raise ValueError("border evidence leaves an empty valid tile region")
        local = values[region.y0 - tile.y0:region.y1 - tile.y0, region.x0 - tile.x0:region.x1 - tile.x0]
        output[region.y0:region.y1, region.x0:region.x1] += local
        count[region.y0:region.y1, region.x0:region.x1] += 1
    if not np.all(count):
        raise ValueError("tile plan leaves source pixels uncovered")
    return output / count
