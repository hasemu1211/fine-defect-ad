"""R1 configuration, raw-score calibration, and public-normal audit primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

LOW_FPR_CLAIM_STATE = "INSUFFICIENT_EVIDENCE_FOR_LOW_FPR_CLAIM"
PROTOCOL_BLOCKED_STATE = "BLOCKED_MISSING_VERIFIED_PROTOCOL_PROVENANCE"
R1_SEED = 607_801_154
R1_SEED_IDENTITY = "fine-defect-ad:R1:EfficientAD-S:anomalib-lib-v2.6.0:3759687e76395c4d6d239552d3bf6d72e003da78"
R1_SEED_IDENTITY_SHA256 = "243a4f42c479e64cedb07d1c7b9eb140c77532c08de948161e019b665f3829ae"
R1_SEED_DERIVATION = "sha256(identity) first 32 bits masked to 31-bit"
_REQUIRED_CONFIG = frozenset({
    "image_size", "batch_size", "model_size", "learning_rate", "weight_decay",
    "max_steps", "max_epochs", "normalization", "seeds", "seed_provenance", "pilot_steps",
})


@dataclass(frozen=True)
class EfficientADSConfig:
    image_size: tuple[int, int]
    batch_size: int
    model_size: str
    learning_rate: float
    weight_decay: float
    max_steps: int
    max_epochs: int
    normalization: None | bool | str
    seeds: tuple[int, ...]
    seed_provenance: Mapping[str, Any]
    pilot_steps: int


def _config_mapping(config: EfficientADSConfig | Mapping[str, Any]) -> dict[str, Any]:
    data = asdict(config) if isinstance(config, EfficientADSConfig) else dict(config)
    forbidden = [str(key) for key in data if "test" in str(key).lower() or "target" in str(key).lower()]
    if forbidden:
        raise ValueError(f"target/test-derived config fields are forbidden: {sorted(forbidden)}")
    unknown = set(data) - _REQUIRED_CONFIG
    missing = _REQUIRED_CONFIG - set(data)
    if unknown or missing:
        raise ValueError(f"R1 config fields mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
    return data


def validate_efficientad_s_config(config: EfficientADSConfig | Mapping[str, Any]) -> EfficientADSConfig:
    """Enforce the fixed R1 EfficientAD-S protocol; no tuning input is accepted."""
    data = _config_mapping(config)
    result = EfficientADSConfig(**data)
    if result.image_size != (256, 256) or result.batch_size != 1 or result.model_size != "small":
        raise ValueError("R1 requires EfficientAD-S at 256x256 with batch_size=1")
    if result.learning_rate != 1e-4 or result.weight_decay != 1e-5:
        raise ValueError("R1 requires learning_rate=1e-4 and weight_decay=1e-5")
    if result.max_steps != 70_000 or result.max_epochs != 1_000 or result.pilot_steps != 1_000:
        raise ValueError("R1 requires max_steps=70000, max_epochs=1000, and pilot_steps=1000")
    if result.normalization not in (None, False):
        raise ValueError("R1 raw-score protocol forbids Normalize")
    if result.seeds != (R1_SEED,) or not _valid_seed_provenance(result.seed_provenance):
        raise ValueError("R1 requires the identity-derived seed and serialized provenance")
    return result


def _values(value: Any) -> Iterable[float]:
    """Flatten Python sequences and numpy-like arrays without depending on numpy."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("raw map must be numeric or a nested numeric sequence")
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _values(item)
        return
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("raw scores must be finite")
    yield number


def _valid_seed_provenance(provenance: Mapping[str, Any]) -> bool:
    return dict(provenance) == {
        "status": "VERIFIED", "upstream_seed_status": "ABSENT",
        "identity": R1_SEED_IDENTITY, "identity_sha256": R1_SEED_IDENTITY_SHA256,
        "derivation": R1_SEED_DERIVATION, "seed": R1_SEED,
    }


def protocol_provenance_status(provenance: Mapping[str, Any]) -> dict[str, str]:
    """Expose a blocking state instead of inventing an unverified comparator."""
    required = {"source_locator", "source_sha256"}
    status = provenance.get("comparator_status", provenance.get("status"))
    if not required <= set(provenance) or status != "VERIFIED":
        return {"claim_state": PROTOCOL_BLOCKED_STATE}
    digest = provenance["source_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        return {"claim_state": PROTOCOL_BLOCKED_STATE}
    return {"claim_state": "VERIFIED"}


def _formula_is_verified(provenance: Mapping[str, Any]) -> bool:
    return provenance.get("formula_status") == "VERIFIED" and bool(provenance.get("formula_source_locator"))


def _comparator(provenance: Mapping[str, Any]) -> str | None:
    if protocol_provenance_status(provenance)["claim_state"] != "VERIFIED":
        return None
    comparator = provenance.get("comparator")
    if comparator not in {">", ">="}:
        raise ValueError("verified provenance comparator must be '>' or '>='")
    return comparator


@dataclass(frozen=True)
class RawThreshold:
    value: float
    comparator: str | None
    provenance: Mapping[str, Any]

    def is_positive(self, score: float) -> bool:
        if self.comparator is None:
            raise ValueError("pixel/image decisions are blocked pending verified comparator provenance")
        return score > self.value if self.comparator == ">" else score >= self.value


def calibrate_raw_threshold(validation_maps: Iterable[Any], provenance: Mapping[str, Any]) -> RawThreshold:
    """Calculate one raw pixel threshold (mean + 3 population standard deviations)."""
    if not _formula_is_verified(provenance):
        raise ValueError("raw threshold formula requires verified provenance")
    comparator = _comparator(provenance)
    scores = [score for raw_map in validation_maps for score in _values(raw_map)]
    if not scores:
        raise ValueError("validation maps cannot be empty")
    mean = math.fsum(scores) / len(scores)
    variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
    return RawThreshold(mean + 3 * math.sqrt(variance), comparator, MappingProxyType(dict(provenance)))


def raw_image_score(raw_map: Any) -> float:
    """The image score is the maximum unmodified pixel score."""
    try:
        return max(_values(raw_map))
    except ValueError as exc:
        raise ValueError("raw map cannot be empty") from exc


def _pixel_decisions(raw_map: Any, threshold: RawThreshold) -> Any:
    values = raw_map.tolist() if hasattr(raw_map, "tolist") else raw_map
    if isinstance(values, (list, tuple)):
        return tuple(_pixel_decisions(value, threshold) for value in values)
    return threshold.is_positive(float(values))


def threshold_raw_map(raw_map: Any, threshold: RawThreshold) -> tuple[Any, Any, bool, float]:
    """Return the original map, shape-preserving pixel decisions, and max-pixel verdict."""
    decisions = _pixel_decisions(raw_map, threshold)
    score = raw_image_score(raw_map)
    return raw_map, decisions, threshold.is_positive(score), score


def clopper_pearson_upper(positives: int, total: int, confidence: float) -> float:
    """Exact one-sided binomial upper confidence bound via monotone bisection."""
    if not isinstance(positives, int) or not isinstance(total, int) or not 0 <= positives <= total or total < 1:
        raise ValueError("require integer counts with 0 <= positives <= total and total >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if positives == total:
        return 1.0
    alpha = 1.0 - confidence
    if positives == 0:
        return 1.0 - alpha ** (1.0 / total)

    def cdf(probability: float) -> float:
        return math.fsum(math.comb(total, index) * probability ** index * (1 - probability) ** (total - index)
                         for index in range(positives + 1))
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass(frozen=True)
class PublicNormalAudit:
    normal_count: int
    positives: int
    empirical_fpr: float
    clopper_pearson_upper: float
    confidence: float
    minimum_normal_samples: int
    sample_size_sufficient: bool
    claim_state: str = LOW_FPR_CLAIM_STATE


def audit_testpub_normal(raw_maps: Iterable[Any], threshold: RawThreshold, *, confidence: float,
                         minimum_normal_samples: int) -> PublicNormalAudit:
    """Audit TESTpub normal images only; it always declines a low-FPR claim."""
    if minimum_normal_samples < 1:
        raise ValueError("minimum_normal_samples must be positive")
    maps = tuple(raw_maps)
    total = len(maps)
    if total < 1:
        raise ValueError("TESTpub-normal audit requires at least one normal image")
    positives = sum(threshold.is_positive(raw_image_score(raw_map)) for raw_map in maps)
    return PublicNormalAudit(total, positives, positives / total,
                             clopper_pearson_upper(positives, total, confidence), confidence,
                             minimum_normal_samples, total >= minimum_normal_samples)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class R1Contract:
    config: tuple[tuple[str, Any], ...]
    config_hash: str
    test_identity_hash: str
    contract_hash: str


def freeze_r1_contract(config: EfficientADSConfig | Mapping[str, Any], test_identities: Iterable[str]) -> R1Contract:
    """Freeze validated config separately from test identities; results are deliberately absent."""
    frozen_config = _freeze(asdict(validate_efficientad_s_config(config)))
    identities = tuple(sorted(str(identity) for identity in test_identities))
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("test identities must be nonempty and unique")
    config_hash, identity_hash = _hash(frozen_config), _hash(identities)
    return R1Contract(frozen_config, config_hash, identity_hash, _hash((frozen_config, identities)))
