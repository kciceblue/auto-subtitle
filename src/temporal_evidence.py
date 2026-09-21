"""Pure temporal grouping of fallible observations; no models, I/O or text edits.

Groups describe partial acoustic intervals, never competing full-owner readings.
All coordinates are absolute and half-open. An owner label is supplied metadata,
not an inference that a crop's words belong to that subtitle owner.
"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import json
import re
from typing import Any, Protocol


class SourceRow(Protocol):
    index: int
    ts_line: str
    text: str


VERSION = "temporal-evidence-1"
_STAMP = re.compile(r"([0-9]{2}):([0-5][0-9]):([0-5][0-9]),([0-9]{3})")
_OBSERVATION_KEYS = {
    "observation_id", "owner_id", "text", "crop_start_frame", "crop_end_frame", "sample_rate",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_copy(value: Any) -> Any:
    """Reject lossy JSON conversions and copy all supplied metadata unchanged."""
    def check(item: Any) -> None:
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "JSON keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        else:
            _require(type(item) in (str, int, float, bool, type(None)), "Invalid JSON value")

    try:
        check(value)
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return deepcopy(value)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Input must be finite, lossless UTF-8 JSON") from exc


def _milliseconds(stamp: str) -> int:
    match = _STAMP.fullmatch(stamp)
    _require(match is not None, "Invalid source timestamp")
    assert match is not None
    hours, minutes, seconds, millis = map(int, match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _ratio(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _intersection(start: int, end: int, rate: int,
                  owner: tuple[int, int]) -> dict[str, Any] | None:
    left = max(Fraction(start, rate), Fraction(owner[0], 1000))
    right = min(Fraction(end, rate), Fraction(owner[1], 1000))
    if left >= right:
        return None
    return {
        "start_seconds": _ratio(left), "end_seconds": _ratio(right),
        "duration_seconds": _ratio(right - left),
        "start_frame": _ratio(left * rate), "end_frame": _ratio(right * rate),
    }


def build_temporal_groups(
    source_rows: Iterable[SourceRow | dict[str, Any]],
    observations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return chronological groups and lossless copies of source/observations.

    Source rows are SrtBlock-compatible dataclass instances or dictionaries with
    index/ts_line/text. IDs need not be contiguous. Every observation needs a
    unique nonblank string ID, a known integer owner ID, string text, and integer
    sample bounds/rate. Additional JSON fields and empty texts are preserved.

    Only identical (owner_id, sample_rate, start_frame, end_frame) tuples group.
    Different sample grids remain separate even if elapsed-time endpoints match.
    Groups sort by exact start/end time, owner ID and sample rate; observations
    within a group sort by their unchanged IDs. Unobserved source owners remain
    in source_rows. A disjoint or merely touching owner intersection is null.
    Fractional frame coordinates denote exact boundaries, not rounded sample
    indices, word timestamps, or evidence that any particular word was spoken.
    Invalid inputs raise ValueError; input objects are never modified.
    """
    source: list[dict[str, Any]] = []
    owners: dict[int, tuple[int, int]] = {}
    for value in source_rows:
        row = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
        _require(type(row) is dict and set(row) == {"index", "ts_line", "text"},
                 "Invalid source row fields")
        _require(type(row["index"]) is int and row["index"] > 0, "Invalid source owner ID")
        _require(row["index"] not in owners, "Duplicate source owner ID")
        _require(type(row["ts_line"]) is str and type(row["text"]) is str, "Invalid source text")
        bounds = row["ts_line"].split(" --> ")
        _require(len(bounds) == 2, "Invalid source interval")
        start, end = map(_milliseconds, bounds)
        _require(start < end, "Source interval must be positive")
        owners[row["index"]] = (start, end)
        source.append(_json_copy(row))

    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for item in observations:
        _require(type(item) is dict and _OBSERVATION_KEYS <= set(item),
                 "Missing observation fields")
        identifier = item["observation_id"]
        _require(type(identifier) is str and bool(identifier.strip()), "Invalid observation ID")
        _require(identifier not in seen, "Duplicate observation ID")
        _require(type(item["owner_id"]) is int and item["owner_id"] in owners,
                 "Unknown observation owner")
        _require(type(item["text"]) is str, "Invalid observation text")
        _require(all(type(item[key]) is int for key in (
            "crop_start_frame", "crop_end_frame", "sample_rate")), "Sample geometry must be integer")
        start, end, rate = item["crop_start_frame"], item["crop_end_frame"], item["sample_rate"]
        _require(0 <= start < end and rate > 0, "Invalid observation interval")
        key = (item["owner_id"], rate, start, end)
        grouped.setdefault(key, []).append(_json_copy(item))
        seen.add(identifier)

    groups = []
    order = lambda key: (Fraction(key[2], key[1]), Fraction(key[3], key[1]), key[0], key[1])
    for owner_id, rate, start, end in sorted(grouped, key=order):
        owner = owners[owner_id]
        groups.append({
            "owner_id": owner_id,
            "scope": "partial_interval",
            "full_owner_competing_readings": False,
            "crop_start_frame": start, "crop_end_frame": end, "sample_rate": rate,
            "owner_interval_ms": {"start": owner[0], "end": owner[1]},
            "owner_intersection": _intersection(start, end, rate, owner),
            "observations": sorted(grouped[(owner_id, rate, start, end)],
                                   key=lambda item: item["observation_id"]),
        })
    return {
        "version": VERSION,
        "interval_convention": "absolute_half_open",
        "grouping_basis": "owner_and_exact_sample_interval",
        "authority": {"word_ownership_inferred": False, "acoustic_truth_verified": False},
        "source_rows": sorted(source, key=lambda row: (*owners[row["index"]], row["index"])),
        "groups": groups,
    }
