"""Strict draft JSON: request-local IDs, preserved guards and resumable raw caches."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import TranslateConfig
from src.entities import EntityPlan, EntityReplacement, EntityRowPlan
from src.evidence import CueEvidence
from src import quality
from src.translate import SrtBlock
from src.workflow_state import fingerprint


class StructuredTranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "requests.json"
        self.source = [SrtBlock(i, f"00:00:{i:02d},000 --> 00:00:{i+1:02d},000", "元気です。")
                       for i in range(1, 10)]
        self.evidence = [CueEvidence(b.index, b.text, "A", b.text, b.text) for b in self.source]
        self.cfg = TranslateConfig(structured_translation=True, stage="translation", max_tokens=4096)

    def query(self, ids=(4, 9), config=None, entity_plan=None):
        return quality.query(list(ids), self.source, self.evidence, config or self.cfg,
                             quality.TRANSLATE, "translation", self.cache, entity_plan=entity_plan)

    def test_sparse_global_ids_are_requested_as_exact_local_json_keys(self):
        payload = {"model": "local-test", "temperature": .3}
        with patch("src.quality.call_llm", return_value='{"1":"很好。","2":"谢谢。"}') as call:
            result = self.query(config=replace(self.cfg, extra_payload=payload))
        self.assertEqual(result, {4: ["很好。"], 9: ["谢谢。"]})
        body, instruction, cfg = call.call_args.args
        schema = cfg.extra_payload["response_format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"1", "2"})
        self.assertEqual(schema["required"], ["1", "2"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["1"], {"type": "string", "minLength": 1})
        self.assertIn("本次仅处理这些ID: 1, 2", body)
        self.assertNotIn("Output [N] translation", instruction)
        self.assertEqual(payload, {"model": "local-test", "temperature": .3})
        self.assertEqual(cfg.max_tokens, 4096)

    def test_invalid_shape_never_has_positional_or_numbered_fallback(self):
        replies = [
            '{"1":"很好。"', '{"1":"很好。"}', '{"1":"很好。","2":"谢谢。","3":"好。"}',
            '{"1":"很好。","1":"谢谢。","2":"好。"}', '{"4":"很好。","9":"谢谢。"}',
            '["很好。","谢谢。"]', 'null', '{"1":null,"2":"谢谢。"}',
            '{"1":NaN,"2":"谢谢。"}', '{"1":{"text":"很好。"},"2":"谢谢。"}',
            '{"1":"  ","2":"谢谢。"}',
            chr(96)*3 + 'json\n{"1":"很好。","2":"谢谢。"}\n' + chr(96)*3,
            '[1] 很好。\n[2] 谢谢。', '1 很好。\n2 谢谢。',
        ]
        for value in ["很好。\n[2] 错误。", "很好。\r", "很好。\x00", "很好。\u2028"]:
            replies.append(json.dumps({"1": value, "2": "谢谢。"}))
        for response in replies:
            with self.subTest(response=response):
                self.assertEqual(quality.parse_structured_translation(
                    response, [1, 2], self.evidence[:2]), {})

    def test_existing_content_and_marker_guards_still_apply(self):
        for value in ["⟦E0001⟧。", "[2] 很好。", "好" * 1000]:
            with self.subTest(value=value):
                self.assertEqual(quality.parse_structured_translation(
                    json.dumps({"1": value}), [1], self.evidence[:1]), {})
        self.assertEqual(quality.parse_structured_translation(
            '{"1":"很好。"}', [1], self.evidence[:1]), {1: ["很好。"]})

    def test_partial_semantic_failure_retries_only_remaining_local_id(self):
        replies = ['{"1":"⟦E0001⟧。","2":"很好。"}', '{"1":"谢谢。"}']
        with patch("src.quality.call_llm", side_effect=replies) as call:
            result = self.query()
        self.assertEqual(result, {4: ["谢谢。"], 9: ["很好。"]})
        self.assertEqual(call.call_args_list[1].args[2].extra_payload["response_format"]["schema"]["required"], ["1"])
        records = [json.loads(line) for line in self.cache.with_suffix(".rejections.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["format"], "structured-translation")
        self.assertEqual(records[0]["valid_global_ids"], [9])
        with patch("src.quality.call_llm") as call:
            self.assertEqual(self.query(), result)
        call.assert_not_called()

    def test_missing_key_retries_complete_shape_instead_of_accepting_partial_json(self):
        with patch("src.quality.call_llm", side_effect=[
            '{"1":"很好。"}', '{"1":"很好。","2":"谢谢。"}']) as call:
            self.assertEqual(self.query(), {4: ["很好。"], 9: ["谢谢。"]})
        self.assertEqual(call.call_args_list[1].args[2].extra_payload["response_format"]["schema"]["required"], ["1", "2"])

    def test_interrupted_request_resumes_from_exact_accepted_raw_response(self):
        raw = '{"1":"⟦E0001⟧。","2":"很好。"}'
        with patch("src.quality.call_llm", side_effect=[raw, RuntimeError("connection interrupted")]):
            with self.assertRaisesRegex(RuntimeError, "connection interrupted"):
                self.query()
        saved = next(iter(json.loads(self.cache.read_text()).values()))
        self.assertEqual(saved["responses"], [{"global_ids": [4, 9], "raw_answer": raw}])
        with patch("src.quality.call_llm", return_value='{"1":"谢谢。"}') as call:
            self.assertEqual(self.query(), {4: ["谢谢。"], 9: ["很好。"]})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[2].extra_payload["response_format"]["schema"]["required"], ["1"])

    def test_corrupted_raw_or_id_scope_cache_is_not_trusted(self):
        good = '{"1":"很好。","2":"谢谢。"}'
        mutations = [
            lambda entry: entry["responses"][0].update(raw_answer='{"1":"很好。","2":"谢谢。"'),
            lambda entry: entry["responses"][0].update(raw_answer='{"1":"很好。","1":"错。","2":"谢谢。"}'),
            lambda entry: entry["responses"][0].update(global_ids=[9, 4]),
            lambda entry: entry["responses"][0].update(global_ids=[4, 9, 9]),
            lambda entry: entry.update(format="wrong-schema"),
            lambda entry: entry.update(trusted_results={"4": ["伪造。"]}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.cache.unlink(missing_ok=True)
                with patch("src.quality.call_llm", return_value=good):
                    self.query()
                saved = json.loads(self.cache.read_text())
                mutate(next(iter(saved.values())))
                self.cache.write_text(json.dumps(saved))
                with patch("src.quality.call_llm", return_value=good) as call:
                    self.assertEqual(self.query(), {4: ["很好。"], 9: ["谢谢。"]})
                self.assertEqual(call.call_count, 1)

    def test_plain_cache_cannot_satisfy_structured_request_and_schema_version_binds(self):
        with patch("src.quality.call_llm", return_value="[1] 很好。\n[2] 谢谢。") as call:
            self.query(config=replace(self.cfg, structured_translation=False))
        self.assertNotIn("response_format", call.call_args.args[2].extra_payload or {})
        with patch("src.quality.call_llm", return_value='{"1":"很好。","2":"谢谢。"}') as call:
            self.query()
            with patch("src.quality.STRUCTURED_TRANSLATION_VERSION", "structured-translation-next"):
                self.query()
        self.assertEqual(call.call_count, 2)
        self.assertEqual(len(json.loads(self.cache.read_text())), 3)

    def test_entity_restoration_replays_raw_cache_with_original_marker_contract(self):
        self.source[0] = replace(self.source[0], text="ユキ。")
        self.evidence[0] = CueEvidence(1, "ユキ。", "A", "ユキ。", "ユキ。")
        marker = EntityReplacement("E0001", 0, 2, "ユキ", "⟦E0001⟧", "g1", "小雪", "ユキ → 小雪")
        rows = {b.index: EntityRowPlan(b.index, b.ts_line, b.text, b.text, ()) for b in self.source}
        rows[1] = EntityRowPlan(1, self.source[0].ts_line, "ユキ。", "⟦E0001⟧。", (marker,))
        plan = EntityPlan([replace(b, text=rows[b.index].marked_source) for b in self.source],
            rows, [], {}, "context", "entity-key", {b.index: fingerprint(b.text) for b in self.source}, "metadata")
        raw = '{"1":"⟦E0001⟧。","2":"很好。"}'
        with patch("src.quality.call_llm", return_value=raw) as call:
            expected = self.query(ids=(1, 3), entity_plan=plan)
            self.assertEqual(self.query(ids=(1, 3), entity_plan=plan), expected)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(expected, {1: ["小雪。"], 3: ["很好。"]})
        cached = next(iter(json.loads(self.cache.read_text()).values()))
        self.assertEqual(cached["responses"][0]["raw_answer"], raw)

    def test_structured_qa_remains_separate_and_unchanged(self):
        cfg = replace(self.cfg, structured_qa=True)
        with patch("src.quality.call_llm", return_value='{"1":{"status":"OK","reason":""}}') as call:
            result = quality.query([4], self.source, self.evidence, cfg,
                                   quality.DIAGNOSE, "qa", self.cache)
        self.assertEqual(result, {4: ["OK"]})
        schema = call.call_args.args[2].extra_payload["response_format"]["schema"]
        self.assertEqual(schema, quality.qa_schema([1]))
        self.assertEqual(call.call_args.args[2].max_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
