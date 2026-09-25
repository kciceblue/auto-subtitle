"""Reversible removal of operational handles before fresh writer generation.

Only three top-level fields move to a local sidecar. All text, owner IDs,
recognizer identities, acquisition states and scope metadata stay unchanged.
No model output, historical receipt or input record is edited in place.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

VERSION = "fresh-six-operational-provenance-sidecar-2"
REMOVED_FIELDS = ("observation_id", "receipt_path", "receipt_sha256")
_ROOT_KEYS = {"version", "removed_fields", "source_records_sha256", "model_records_sha256", "records"}
_ROW_KEYS = {"record_index", "source_sha256", "model_sha256", "removed"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(value: Any) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "Nonfinite record value")
        return
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("Record is not valid UTF-8") from exc
        return
    if type(value) is list:
        for item in value:
            _json(item)
        return
    _require(type(value) is dict and all(type(key) is str for key in value), "Records must contain actual JSON values")
    for key, item in value.items():
        _json(key)
        _json(item)


def _records(records: list[dict]) -> None:
    _require(type(records) is list and all(type(row) is dict for row in records), "Expected a record list")
    _json(records)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sanitize(records: list[dict]) -> tuple[list[dict], dict]:
    """Return fresh model records and an authenticated, entirely local sidecar."""
    _records(records)
    clean, audit = [], []
    for index, original in enumerate(records):
        row = {key: deepcopy(value) for key, value in original.items() if key not in REMOVED_FIELDS}
        removed = {key: deepcopy(original[key]) for key in REMOVED_FIELDS if key in original}
        clean.append(row)
        audit.append({"record_index": index, "source_sha256": _hash(original),
                      "model_sha256": _hash(row), "removed": removed})
    sidecar = {"version": VERSION, "removed_fields": list(REMOVED_FIELDS),
               "source_records_sha256": _hash(records), "model_records_sha256": _hash(clean), "records": audit}
    _require(restore(clean, sidecar) == records, "Operational projection did not reconstruct original evidence")
    return clean, sidecar


def restore(records: list[dict], sidecar: dict) -> list[dict]:
    """Authenticate the exact projection and reconstruct every original field."""
    _records(records)
    _json(sidecar)
    _require(type(sidecar) is dict and set(sidecar) == _ROOT_KEYS
             and sidecar["version"] == VERSION and sidecar["removed_fields"] == list(REMOVED_FIELDS),
             "Operational sidecar policy changed")
    audit = sidecar["records"]
    _require(type(audit) is list and len(audit) == len(records)
             and sidecar["model_records_sha256"] == _hash(records), "Model record sequence changed")
    original = []
    for index, (row, evidence) in enumerate(zip(records, audit)):
        _require(type(evidence) is dict and set(evidence) == _ROW_KEYS
                 and type(evidence["record_index"]) is int and evidence["record_index"] == index,
                 "Operational record association changed")
        removed = evidence["removed"]
        _require(type(removed) is dict and set(removed) <= set(REMOVED_FIELDS)
                 and not set(row).intersection(REMOVED_FIELDS), "Sidecar would replace a semantic or retained field")
        _require(evidence["model_sha256"] == _hash(row), "Model record changed")
        restored = {**deepcopy(row), **deepcopy(removed)}
        _require(evidence["source_sha256"] == _hash(restored), "Original evidence reconstruction changed")
        original.append(restored)
    _require(sidecar["source_records_sha256"] == _hash(original), "Original record sequence changed")
    return original
