"""Operational-field projection checks using synthetic records only."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fresh_writer_inputs import REMOVED_FIELDS, restore, sanitize


def records():
    return [{"owner_id": 1, "text": " 字面原文\n ", "start": 0.0, "end": 1.25,
             "observer": "qwen-instance", "status": "ok", "view": "raw", "core_start": .1,
             "observation_id": "original-1", "receipt_path": "/local/receipt.json", "receipt_sha256": "a" * 64,
             "metadata": {"receipt_path": "nested value remains intact", "unknown": None, "empty": ""}},
            {"owner_id": 2, "text": "", "start": 1.25, "end": 2, "observer": "same-instance",
             "status": "unavailable", "detected_language": None, "language_origin": "automatic",
             "vad_overlap_seconds": 0.0, "duplicate_geometry": False, "receipt_path": None},
            {"owner_id": 3, "text": "same literal", "view": "short", "word_ownership_verified": False}]


class ProjectionTests(unittest.TestCase):
    def test_only_declared_top_level_fields_move_and_every_record_roundtrips(self):
        original = records()
        clean, sidecar = sanitize(original)
        self.assertEqual(restore(clean, sidecar), original)
        self.assertEqual(len(clean), len(original))
        for old, new, audit in zip(original, clean, sidecar["records"]):
            self.assertEqual(new, {key: value for key, value in old.items() if key not in REMOVED_FIELDS})
            self.assertEqual(audit["removed"], {key: value for key, value in old.items() if key in REMOVED_FIELDS})
        self.assertIn("receipt_path", clean[0]["metadata"])
        self.assertEqual(clean[0]["text"], " 字面原文\n ")
        self.assertIsNone(sidecar["records"][1]["removed"]["receipt_path"])
        self.assertNotIn("receipt_path", sidecar["records"][2]["removed"])

    def test_input_and_outputs_share_no_mutable_records(self):
        original = records()
        before = deepcopy(original)
        clean, sidecar = sanitize(original)
        restored = restore(clean, sidecar)
        clean[0]["metadata"]["unknown"] = "changed"
        sidecar["records"][0]["removed"]["observation_id"] = "changed"
        restored[0]["metadata"]["empty"] = "changed"
        self.assertEqual(original, before)

    def test_semantic_metadata_edit_and_row_reordering_are_detected(self):
        for mutation in (lambda rows: rows[0].update(text="altered"),
                         lambda rows: rows[0].update(start=.01),
                         lambda rows: rows[1].update(status="empty"),
                         lambda rows: rows[1].update(detected_language="Japanese"),
                         lambda rows: rows.reverse(),
                         lambda rows: rows.append({"text": "extra"})):
            clean, sidecar = sanitize(records())
            mutation(clean)
            with self.assertRaises(ValueError):
                restore(clean, sidecar)

    def test_removed_values_cannot_be_changed_or_promoted_into_semantic_fields(self):
        for mutation in (lambda audit: audit["records"][0]["removed"].update(text="replacement"),
                         lambda audit: audit["records"][0]["removed"].update(receipt_sha256="b" * 64),
                         lambda audit: audit["records"][0].update(record_index=True),
                         lambda audit: audit["records"].reverse(),
                         lambda audit: audit.update(extra="unexpected")):
            clean, sidecar = sanitize(records())
            mutation(sidecar)
            with self.assertRaises(ValueError):
                restore(clean, sidecar)

    def test_empty_record_lists_and_absent_provenance_remain_valid(self):
        for original in ([], [{}], [{"text": "unchanged", "owner_id": 1}]):
            clean, sidecar = sanitize(original)
            self.assertEqual(clean, original)
            self.assertEqual(restore(clean, sidecar), original)

    def test_non_json_values_are_rejected_without_coercion(self):
        for value in ((1, 2), {1: "integer key"}, float("nan"), float("inf"), "\ud800", b"bytes"):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                sanitize([{"text": value}])


if __name__ == "__main__":
    unittest.main()
