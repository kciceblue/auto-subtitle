"""Offline behavioral regressions; run from the repository root.

    python3 -m unittest discover -s tests -p test_evidence_workflow.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import build_parser
from src import quality, workflow
from src.adjudicate import grade_vote
from src.config import TranslateConfig
from src.evidence import CueEvidence, build_evidence
from src.script_align import align_script
from src.translate import SrtBlock, parse_srt, write_translated_srt
from src.workflow_state import StageState, artifact_path, file_hash


def blocks(texts=("こんにちは", "ありがとう")):
    return [SrtBlock(i, f"00:00:{i * 2:02d},000 --> 00:00:{i * 2 + 1:02d},000", text)
            for i, text in enumerate(texts, 1)]


def evidence(source):
    return [CueEvidence(b.index, b.text, "A", b.text, b.text) for b in source]


def decisions(source):
    return [{"line": b.index, "status": "accepted", "text": b.text, "candidate": "W", "reason": "audio"}
            for b in source]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old_cwd)
        self.addCleanup(self.tmp.cleanup)
        self.cfg = TranslateConfig(chunk_size=24, retries=0, endpoint="http://127.0.0.1:9/unused")
        self.src = blocks()
        self.ev = evidence(self.src)
        self.cache = self.root / "requests.json"

    def test_competing_asr_is_confined_to_source_adjudication(self):
        self.ev[0] = CueEvidence(1, "テニスを切る", "B-", "テニスを切る", "ペニスを切る")
        source_prompt = quality.format_scene([1], self.src, self.ev, self.cfg, candidates=True)
        self.assertIn("N=テニスを切る", source_prompt)
        self.assertIn("Q=ペニスを切る", source_prompt)
        for stage in ("translation", "diagnosis", "verification"):
            text = quality.format_scene([1], self.src, self.ev, quality.mode(self.cfg, stage))
            self.assertIn(self.src[0].text,text)
            self.assertNotIn("Q=",text)
            self.assertNotIn("ペニスを切る",text)
        self.assertEqual(self.ev[0].q,"ペニスを切る")

    def test_phonetic_alignment_does_not_hide_kanji_conflict(self):
        anchor = align_script(["虚勢手術を始めます"], ["去勢手術を始めます"])[0]
        self.assertEqual(anchor.status, "high")
        self.assertTrue(anchor.orthographic_conflict)

    def test_one_missing_asr_is_degraded_not_a_three_way_vote(self):
        self.assertEqual(grade_vote("a", "a", "", "a", "a", "")[0], "D")
        self.assertEqual(grade_vote("a", "", "a", "a", "", "a")[0], "D")

    def test_source_selection_keeps_raw_evidence(self):
        src = blocks(("テニスを切る",))
        ev = [CueEvidence(1, src[0].text, "B-", src[0].text, "ペニスを切る")]
        with patch.object(quality, "call_llm", return_value="[1] SELECT|Q|独立候选与语境一致") as llm:
            resolved, result = quality.resolve_source(src, ev, self.cfg, self.cache)
        self.assertEqual(resolved[0].text, "ペニスを切る")
        self.assertEqual(src[0].text, "テニスを切る")
        self.assertEqual(result[0]["evidence"]["raw"], "テニスを切る")
        self.assertEqual(result[0]["candidate"], "Q")
        self.assertTrue(llm.call_args.kwargs["with_thinking"])

    def test_hard_conflict_never_enters_resolution_or_repair(self):
        ev = [replace(self.ev[0], script="違う台詞", script_status="mismatch"), self.ev[1]]
        with patch.object(quality, "call_llm") as llm:
            resolved, result = quality.resolve_source(self.src, ev, self.cfg, self.cache)
            llm.assert_not_called()
        self.assertEqual(result[0]["status"], "unresolved")
        with patch.object(quality, "call_llm", return_value="[1] ISSUE|meaning|台本と違う\n[2] OK") as llm:
            final, ledger = quality.selective_qa(resolved, blocks(("你好", "谢谢")), ev, result, self.cfg, self.cache)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(ledger[0]["status"], "unresolved")
        self.assertEqual(final[0].text, "你好")

    def test_c_and_d_are_visible_even_when_qa_says_ok(self):
        ev = [replace(self.ev[0], grade="C"), replace(self.ev[1], grade="D")]
        _, result = quality.resolve_source(self.src, ev, self.cfg, self.cache)
        with patch.object(quality, "call_llm", return_value="[1] OK\n[2] OK"):
            _, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), ev, result, self.cfg, self.cache)
        self.assertEqual([r["status"] for r in ledger], ["unresolved", "unresolved"])

    def test_low_confidence_ensemble_source_cannot_be_reconstructed_by_writer(self):
        source = blocks(("あしたえきで待つ。",))
        ev = [CueEvidence(1, source[0].text, "C", "明日駅で待つ。", "あした駅で待ちます。", ensemble=True)]
        with patch.object(quality, "call_llm", return_value="[1] PROPOSE|明日駅で待ちます。|推測") as llm:
            resolved, result = quality.resolve_source(source, ev, self.cfg, self.cache)
        llm.assert_not_called()
        self.assertEqual(resolved[0].text, source[0].text)
        self.assertEqual(resolved[0].ts_line, source[0].ts_line)
        self.assertEqual(result[0]["status"], "unresolved")
        self.assertEqual(result[0]["evidence"]["q"], ev[0].q)
        with patch.object(quality, "call_llm", return_value="[1] OK"):
            _, ledger = quality.selective_qa(resolved, blocks(("明天在车站等。",)), ev, result, self.cfg, self.cache)
        self.assertEqual(ledger[0]["status"], "unresolved")
        self.assertEqual(ledger[0]["target_status"], "accepted")

    def test_ensemble_spelling_conflict_uses_bounded_candidate_selection(self):
        source = blocks(("川井先生です。", "ありがとうございます。"))
        ev = [replace(e, ensemble=True, grade=grade) for e, grade in zip(evidence(source), ("E-音", "A"))]
        ev[0] = replace(ev[0], q="河合先生です。")
        cfg = replace(self.cfg, context_summary="登場人物：河合先生（かわい先生）。")
        with patch.object(quality, "call_llm", return_value="[1] SELECT|Q|名字与资料一致") as llm:
            resolved, result = quality.resolve_source(source, ev, cfg, self.cache)
        self.assertEqual([r["status"] for r in result], ["corrected", "accepted"])
        self.assertEqual(resolved[0].text, ev[0].q)
        self.assertFalse(llm.call_args.kwargs["with_thinking"])
        self.assertEqual(llm.call_args.args[2].max_tokens, 2048)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result[0]["candidate"], "Q")

    def test_low_confidence_candidate_selection_cannot_become_source_approval(self):
        source = blocks(("あしたえきで待つ。",))
        for grade in ("C", "D"):
            for action in ("KEEP|暂用当前候选", "SELECT|Q|完整候选"):
                with self.subTest(grade=grade, action=action):
                    ev = [CueEvidence(1, source[0].text, grade, "明日駅で待つ。", "明日駅で待ちます。", ensemble=True)]
                    with patch.object(quality, "call_llm", return_value=f"[1] {action}") as llm:
                        resolved, result = quality.resolve_source(source, ev, self.cfg, self.root / f"{grade}-{action[0]}.json")
                    llm.assert_not_called()
                    self.assertEqual(result[0]["status"], "unresolved")
                    self.assertEqual(resolved[0].text, source[0].text)

    def test_ensemble_hard_conflict_is_never_reconstructed(self):
        source = blocks(("こんにちは",))
        ev = [replace(evidence(source)[0], ensemble=True, script="違う台詞", script_status="mismatch")]
        with patch.object(quality, "call_llm") as llm:
            resolved, result = quality.resolve_source(source, ev, self.cfg, self.cache)
        llm.assert_not_called()
        self.assertEqual(result[0]["status"], "unresolved")
        self.assertEqual(resolved[0].text, source[0].text)
        self.assertEqual(quality.parse_records("[1] PROPOSE|こんばんは|推測", [1], "resolve", ev), {})

    def test_legacy_low_confidence_source_remains_nonwritable(self):
        ev = [replace(self.ev[0], grade="C"), replace(self.ev[1], grade="D")]
        with patch.object(quality, "call_llm") as llm:
            resolved, result = quality.resolve_source(self.src, ev, self.cfg, self.cache)
        llm.assert_not_called()
        self.assertEqual([r["status"] for r in result], ["unresolved", "unresolved"])
        self.assertEqual([b.text for b in resolved], [b.text for b in self.src])
        self.assertEqual(quality.parse_records("[1] PROPOSE|こんばんは|推測", [1], "resolve", ev), {})

    def test_invalid_or_missing_ensemble_source_review_never_rewrites(self):
        source = blocks(("こんにちは",))
        ev = [replace(evidence(source)[0], ensemble=True, grade="E-音")]
        invalid = ["[1] SELECT|UNKNOWN|理由", "[1] PROPOSE||理由", "[1] PROPOSE|こんばんは|无候选的重写", "无", "[2] KEEP|错号",
                   "[1] PROPOSE|" + "新しい話。" * 200 + "|候选没有这些内容",
                   "[1] PROPOSE|こんにちは[2]さようなら|串行"]
        for index, reply in enumerate(invalid):
            with self.subTest(reply=reply[:60]):
                with patch.object(quality, "call_llm", return_value=reply) as llm:
                    resolved, result = quality.resolve_source(source, ev, self.cfg, self.root / f"invalid-{index}.json")
                self.assertEqual(llm.call_count, 2)
                self.assertEqual(result[0]["status"], "unchecked")
                self.assertEqual(resolved[0].text, source[0].text)
                self.assertNotIn("proposed_text", result[0])

    def test_wrong_duplicate_and_unmarked_ids_never_pass(self):
        self.assertEqual(quality.parse_records("无", [1, 2], "qa", self.ev), {})
        self.assertEqual(quality.parse_records("[99] OK\n[1] OK", [1, 2], "qa", self.ev), {})
        self.assertEqual(quality.parse_records("[1] OK\n[1] UNSURE|x", [1, 2], "qa", self.ev), {})
        self.assertEqual(quality.parse_records("[1] SELECT|MADE_UP|x", [1], "resolve", self.ev), {})

    def test_format_retry_only_requests_missing_cues_and_resume_reuses_them(self):
        with patch.object(quality, "call_llm", side_effect=["[1] 你好", "[1] 谢谢"]) as llm:
            result = quality.translate_scenes(self.src, self.ev, self.cfg, self.cache)
        self.assertEqual([b.text for b in result], ["你好", "谢谢"])
        self.assertIn("[1] 已选源文: ありがとう", llm.call_args_list[1].args[0])
        self.assertNotIn("[2]", llm.call_args_list[1].args[0])
        self.assertIn("[neighbor]", llm.call_args_list[1].args[0])
        with patch.object(quality, "call_llm") as llm:
            quality.translate_scenes(self.src, self.ev, self.cfg, self.cache)
            llm.assert_not_called()

    def test_sparse_global_ids_are_local_in_model_requests_and_global_in_results(self):
        source = blocks(tuple("台詞です。" for _ in range(12)))
        ev = evidence(source)
        with patch.object(quality, "call_llm", return_value="[1] ISSUE|actor|主体が逆\n[2] OK") as llm:
            result = quality.query([4, 9], source, ev, quality.mode(self.cfg, "verification"),
                                   quality.DIAGNOSE, "qa", self.cache)
        self.assertEqual(result, {4: ["ISSUE", "actor", "主体が逆"], 9: ["OK"]})
        body = llm.call_args.args[0]
        self.assertNotIn("[4]", body)
        self.assertNotIn("[9]", body)
        self.assertNotIn("[context:", body)
        self.assertEqual(__import__("re").findall(r"^\[(\d+)\]", body, __import__("re").M), ["1", "2"])

    def test_missing_sparse_response_rebases_retry_and_caches_global_ids(self):
        source = blocks(tuple("台詞です。" for _ in range(8)))
        ev = evidence(source)
        with patch.object(quality, "call_llm", side_effect=["[1] OK", "[1] UNSURE|音が不明"]) as llm:
            result = quality.query([4, 6], source, ev, self.cfg, quality.DIAGNOSE, "qa", self.cache)
        self.assertEqual(result, {4: ["OK"], 6: ["UNSURE", "音が不明"]})
        retry = llm.call_args_list[1].args[0]
        self.assertIn("[1] 已选源文:", retry)
        self.assertNotIn("[2]", retry)
        self.assertIn("[neighbor]", retry)
        saved = next(iter(json.loads(self.cache.read_text()).values()))
        self.assertEqual(set(saved), {"4", "6"})

    def test_out_of_range_local_context_response_cannot_be_applied(self):
        source = blocks(tuple("台詞です。" for _ in range(10)))
        response = "[1] OK\n[2] OK\n[3] ISSUE|context|邻域不应输出"
        with patch.object(quality, "call_llm", return_value=response) as llm:
            result = quality.query([4, 9], source, evidence(source), self.cfg,
                                   quality.DIAGNOSE, "qa", self.cache)
        self.assertEqual(result, {})
        self.assertEqual(llm.call_count, 2)

    def test_cached_global_results_are_revalidated_after_local_remapping(self):
        source = blocks(tuple("台詞です。" for _ in range(8)))
        args = ([4, 6], source, evidence(source), self.cfg, quality.DIAGNOSE, "qa", self.cache)
        with patch.object(quality, "call_llm", return_value="[1] OK\n[2] OK"):
            expected = quality.query(*args)
        with patch.object(quality, "call_llm") as llm:
            self.assertEqual(quality.query(*args), expected)
            llm.assert_not_called()
        saved = json.loads(self.cache.read_text())
        key = next(iter(saved))
        saved[key] = {"1": ["OK"], "6": ["OK"]}  # Local IDs are invalid in the persistent cache.
        self.cache.write_text(json.dumps(saved))
        with patch.object(quality, "call_llm", return_value="[1] OK\n[2] OK") as llm:
            self.assertEqual(quality.query(*args), expected)
            self.assertEqual(llm.call_count, 1)
        self.assertEqual(set(json.loads(self.cache.read_text())[key]), {"4", "6"})

    def test_local_source_selection_validates_the_global_targets_evidence(self):
        source = blocks(tuple("元の台詞。" for _ in range(5)))
        ev = evidence(source)
        ev[3] = replace(ev[3], title_candidates=("別の候補。",))
        with patch.object(quality, "call_llm", return_value="[1] SELECT|TITLE1|标题支持"):
            result = quality.query([4], source, ev, self.cfg, quality.RESOLVE, "resolve", self.cache)
        self.assertEqual(result, {4: ["SELECT", "TITLE1", "标题支持"]})

    def test_structured_qa_maps_sparse_ids_and_limits_only_qa_requests(self):
        source = blocks(tuple("台詞です。" for _ in range(8)))
        original_payload = {"model": "qwen3.8-27b-dflash", "temperature": 0.3,
                            "chat_template_kwargs": {"enable_thinking": True}}
        cfg = replace(self.cfg, structured_qa=True, extra_payload=original_payload)
        reply = json.dumps({"1": {"status": "ISSUE", "reason": "主体が逆"},
                            "2": {"status": "OK", "reason": ""}})
        with patch.object(quality, "call_llm", return_value=reply) as llm:
            result = quality.query([4, 6], source, evidence(source), cfg,
                                   quality.DIAGNOSE, "qa", self.cache)
        self.assertEqual(result, {4: ["ISSUE", "translation", "主体が逆"], 6: ["OK"]})
        request = llm.call_args.args[2]
        schema = request.extra_payload["response_format"]["schema"]
        self.assertEqual(schema["required"], ["1", "2"])
        self.assertEqual(set(schema["properties"]), {"1", "2"})
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["properties"]["1"]["additionalProperties"])
        self.assertEqual(request.extra_payload["model"], original_payload["model"])
        self.assertEqual(request.max_tokens, 2048)
        self.assertFalse(request.extra_payload["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(original_payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("response_format", original_payload)
        self.assertFalse(llm.call_args.kwargs["with_thinking"])
        with patch.object(quality, "call_llm", return_value="[1] 你好") as llm:
            quality.query([4], source, evidence(source), cfg, quality.TRANSLATE,
                          "translation", self.cache)
        self.assertIs(llm.call_args.args[2], cfg)
        with patch.object(quality, "call_llm", return_value="[1] OK") as llm:
            quality.query([4], source, evidence(source), self.cfg, quality.DIAGNOSE,
                          "qa", self.cache)
        self.assertIs(llm.call_args.args[2], self.cfg)

    def test_structured_qa_rejects_duplicate_missing_extra_and_unbounded_results(self):
        bad = [
            '{"1":{"status":"OK","reason":""},"1":{"status":"ISSUE","reason":"x"}}',
            '{"1":{"status":"ISSUE","status":"OK","reason":"x"}}',
            '{"1":{"status":"OK","reason":"","reason":""}}',
            '{"1":{"status":"OK","reason":""}',
            '```json\n{"1":{"status":"OK","reason":""}}\n```',
            '{"2":{"status":"OK","reason":""}}',
            '{"01":{"status":"OK","reason":""}}',
            '{"1":{"status":"OK","reason":""},"neighbor":{"status":"OK","reason":""}}',
            '{"1":{"status":"OK","reason":"","extra":1}}',
            '{"1":{"status":"OK"}}',
            '{"1":{"status":"PASS","reason":""}}',
            '{"1":{"status":true,"reason":""}}',
            '{"1":{"status":"OK","reason":NaN}}',
            '{"1":{"status":"OK","reason":null}}',
            '{"1":{"status":"ISSUE","reason":"  "}}',
            '{"1":{"status":"UNSURE","reason":""}}',
            json.dumps({"1": {"status": "ISSUE", "reason": "x" * 121}}),
            json.dumps({"1": {"status": "ISSUE", "reason": "x\n[2] OK"}}),
            '[]', '{}', 'null',
        ]
        for reply in bad:
            with self.subTest(reply=reply):
                self.assertEqual(quality.parse_structured_qa(reply, [1]), {})
        self.assertEqual(quality.parse_structured_qa('{"1":{"status":"OK","reason":""}}', [1, 2]), {})

    def test_structured_failure_stays_unchecked_and_retains_rejected_raw_answer(self):
        cfg = replace(self.cfg, structured_qa=True, extra_payload={"test_secret": "not-for-audit"})
        reply = '{"1":{"status":"OK","reason":""}}'  # Required second ID is absent.
        with patch.object(quality, "call_llm", return_value=reply) as llm:
            final, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), self.ev,
                                               decisions(self.src), cfg, self.cache)
        self.assertEqual(llm.call_count, 2)
        self.assertEqual([r["status"] for r in ledger], ["unchecked", "unchecked"])
        self.assertEqual([b.text for b in final], ["你好", "谢谢"])
        audit_text = self.cache.with_suffix(".rejections.jsonl").read_text()
        audit = [json.loads(line) for line in audit_text.splitlines()]
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["global_ids"], [1, 2])
        self.assertEqual(audit[0]["stage"], "diagnosis")
        self.assertEqual(audit[0]["raw_answer"], reply)
        self.assertNotIn("not-for-audit", audit_text)

    def test_structured_verified_target_repair_keeps_source_blocker(self):
        source = blocks(("私が君を殺す。", "やめて。"))
        ev = evidence(source)
        ev[0] = replace(ev[0], grade="C", ensemble=True)
        chosen = decisions(source)
        chosen[0].update(status="unresolved", reason="Audio disagreement")
        draft = blocks(("你去死吧。", "住手。"))
        cfg = replace(self.cfg, structured_qa=True)
        diagnosis = json.dumps({"1": {"status": "ISSUE", "reason": "主体と行為が変わった"},
                                "2": {"status": "OK", "reason": ""}})
        verified = json.dumps({"1": {"status": "OK", "reason": ""},
                               "2": {"status": "OK", "reason": ""}})
        with patch.object(quality, "call_llm", side_effect=[diagnosis, "[1] FIX|我要杀了你。", verified]) as llm:
            final, ledger = quality.selective_qa(source, draft, ev, chosen, cfg, self.cache)
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(final[0].text, "我要杀了你。")
        self.assertEqual(ledger[0]["target_status"], "corrected")
        self.assertEqual(ledger[0]["status"], "unresolved")
        self.assertNotIn("response_format", llm.call_args_list[1].args[2].extra_payload)
        self.assertIn("response_format", llm.call_args_list[2].args[2].extra_payload)
        hard = replace(ev[0], grade="A", script="別の台詞", script_status="mismatch")
        with patch.object(quality, "call_llm", return_value=diagnosis) as llm:
            final, ledger = quality.selective_qa(source, draft, [hard, ev[1]], chosen, cfg,
                                               self.root / "hard.json")
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(final[0].text, draft[0].text)
        self.assertEqual(ledger[0]["status"], "unresolved")

    def test_structured_global_cache_revalidates_and_missing_cache_rebases_schema(self):
        source = blocks(tuple("台詞です。" for _ in range(8)))
        cfg = replace(self.cfg, structured_qa=True)
        args = ([4, 6], source, evidence(source), cfg, quality.DIAGNOSE, "qa", self.cache)
        reply = json.dumps({"1": {"status": "OK", "reason": ""},
                            "2": {"status": "UNSURE", "reason": "音が不明"}})
        with patch.object(quality, "call_llm", return_value=reply):
            expected = quality.query(*args)
        saved = json.loads(self.cache.read_text())
        key = next(iter(saved))
        self.assertEqual(set(saved[key]), {"4", "6"})
        with patch.object(quality, "call_llm") as llm:
            self.assertEqual(quality.query(*args), expected)
            llm.assert_not_called()
        del saved[key]["6"]
        self.cache.write_text(json.dumps(saved))
        with patch.object(quality, "call_llm", return_value='{"1":{"status":"UNSURE","reason":"音が不明"}}') as llm:
            self.assertEqual(quality.query(*args), expected)
        schema = llm.call_args.args[2].extra_payload["response_format"]["schema"]
        self.assertEqual(schema["required"], ["1"])
        self.assertIn("[1] 已选源文:", llm.call_args.args[0])
        for damaged in ({"1": ["OK"], "6": ["OK"]},
                        {"4": ["ISSUE", "translation", "bad\n[6] OK"]}):
            saved[key] = damaged
            self.cache.write_text(json.dumps(saved))
            with patch.object(quality, "call_llm", return_value=reply) as llm:
                self.assertEqual(quality.query(*args), expected)
                self.assertEqual(llm.call_count, 1)

    def test_structured_qa_flag_is_default_off_and_invalidates_workflow_cache(self):
        args = self.args_and_media(names=("track01",))
        self.assertFalse(args.structured_qa)
        self.assertFalse(TranslateConfig().structured_qa)
        self.assertFalse(workflow.collect_jobs(args)[0].config.structured_qa)
        enabled = build_parser().parse_args(["pipeline", "--structured-qa"])
        self.assertTrue(enabled.structured_qa)
        disabled = build_parser().parse_args(["pipeline", "--structured-qa", "--no-structured-qa"])
        self.assertFalse(disabled.structured_qa)
        args.structured_qa = True
        cfg = workflow.collect_jobs(args)[0].config
        self.assertTrue(cfg.structured_qa)
        self.assertNotEqual(workflow.configuration_key(cfg),
                            workflow.configuration_key(replace(cfg, structured_qa=False)))

    def test_outage_does_not_trigger_format_retry_tree(self):
        with patch.object(quality, "call_llm", side_effect=RuntimeError("offline")) as llm:
            with self.assertRaises(RuntimeError):
                quality.translate_scenes(self.src, self.ev, replace(self.cfg, chunk_size=1), self.cache)
        self.assertEqual(llm.call_count, 1)

    def test_all_ok_qa_does_not_regenerate_subtitles(self):
        with patch.object(quality, "call_llm", return_value="[1] OK\n[2] OK") as llm:
            _, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), self.ev,
                                              decisions(self.src), self.cfg, self.cache)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual([r["status"] for r in ledger], ["accepted", "accepted"])

    def test_missing_qa_coverage_remains_unchecked(self):
        with patch.object(quality, "call_llm", side_effect=["[1] OK", "没有问题"]):
            final, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), self.ev,
                                                  decisions(self.src), self.cfg, self.cache)
        self.assertEqual(ledger[1]["status"], "unchecked")
        complete = quality.write_quality_result(Path("main.zh.srt"), final, ledger,
                                                Path("REVIEW-main.zh.md"), Path("main.quality.json"))
        self.assertFalse(complete)
        self.assertIn("未完成", Path("REVIEW-main.zh.md").read_text())

    def test_patch_is_reverted_if_neighbor_does_not_pass(self):
        replies = ["[1] ISSUE|meaning|wrong greeting\n[2] OK", "[1] FIX|您好",
                   "[1] OK\n[2] ISSUE|shift|neighbor meaning lost"]
        with patch.object(quality, "call_llm", side_effect=replies):
            final, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), self.ev,
                                                  decisions(self.src), self.cfg, self.cache)
        self.assertEqual(final[0].text, "你好")
        self.assertEqual(ledger[0]["status"], "unresolved")
        self.assertEqual(ledger[1]["status"], "unresolved")

    def test_verified_patch_is_the_only_rewrite(self):
        replies = ["[1] ISSUE|meaning|wrong greeting\n[2] OK", "[1] FIX|您好", "[1] OK\n[2] OK"]
        with patch.object(quality, "call_llm", side_effect=replies):
            final, ledger = quality.selective_qa(self.src, blocks(("你好", "谢谢")), self.ev,
                                                  decisions(self.src), self.cfg, self.cache)
        self.assertEqual([b.text for b in final], ["您好", "谢谢"])
        self.assertEqual(ledger[0]["status"], "corrected")

    def test_verified_target_repair_does_not_resolve_source_uncertainty(self):
        source = blocks(("私が彼を起こします。",))
        draft = blocks(("他会叫醒我。",))
        ev = [replace(evidence(source)[0], grade="C")]
        for source_status in ("unresolved", "unchecked"):
            with self.subTest(source_status=source_status):
                source_decisions = decisions(source)
                source_decisions[0].update(status=source_status, reason="Audio candidate disagreement")
                replies = ["[1] ISSUE|actor|The subject and object are reversed",
                           "[1] FIX|我会叫醒他。", "[1] OK"]
                with patch.object(quality, "call_llm", side_effect=replies) as llm:
                    final, ledger = quality.selective_qa(
                        source, draft, ev, source_decisions, self.cfg,
                        self.root / f"{source_status}.json")
                self.assertEqual(llm.call_count, 3)
                self.assertEqual(final[0].text, "我会叫醒他。")
                self.assertEqual(source[0].text, "私が彼を起こします。")
                self.assertEqual(ledger[0]["status"], source_status)
                self.assertEqual(ledger[0]["target_status"], "corrected")
                self.assertEqual(ledger[0]["before"], draft[0].text)
                self.assertEqual(ledger[0]["source_decision"], source_decisions[0])
                self.assertIn("Audio candidate disagreement", ledger[0]["reason"])

    def test_uncertain_source_repair_still_requires_neighbor_verification(self):
        source_decisions = decisions(self.src)
        source_decisions[0].update(status="unresolved", reason="Unclear audio")
        ev = [replace(self.ev[0], grade="C"), self.ev[1]]
        replies = ["[1] ISSUE|meaning|wrong greeting\n[2] OK", "[1] FIX|您好",
                   "[1] OK\n[2] ISSUE|shift|neighbor meaning lost"]
        with patch.object(quality, "call_llm", side_effect=replies):
            final, ledger = quality.selective_qa(
                self.src, blocks(("你好", "谢谢")), ev, source_decisions, self.cfg, self.cache)
        self.assertEqual(final[0].text, "你好")
        self.assertEqual(ledger[0]["status"], "unresolved")
        self.assertEqual(ledger[0]["target_status"], "unresolved")
        self.assertIn("Repair reverted", ledger[0]["target_reason"])
        self.assertIn("Unclear audio", ledger[0]["reason"])
        self.assertEqual(ledger[1]["status"], "unresolved")

    def test_missing_target_verification_remains_unchecked_with_source_doubt(self):
        source = blocks(("こんにちは",))
        source_decisions = decisions(source)
        source_decisions[0].update(status="unresolved", reason="Unclear audio")
        replies = ["[1] ISSUE|meaning|wrong greeting", "[1] FIX|您好", "无", "无"]
        with patch.object(quality, "call_llm", side_effect=replies):
            final, ledger = quality.selective_qa(
                source, blocks(("你好",)), evidence(source), source_decisions, self.cfg, self.cache)
        self.assertEqual(final[0].text, "你好")
        self.assertEqual(ledger[0]["status"], "unchecked")
        self.assertEqual(ledger[0]["target_status"], "unchecked")
        self.assertIn("Unclear audio", ledger[0]["reason"])

    def test_scene_context_is_deduplicated_and_not_writable(self):
        text = quality.format_scene([2], self.src, self.ev, self.cfg)
        self.assertEqual(text.count("[context:1]"), 1)
        self.assertNotIn("[1]", text)
        self.assertEqual(quality.scene_groups(self.src, replace(self.cfg, scene_max_chars=5)), [[1], [2]])

    def test_stage_state_invalidates_changed_inputs_and_outputs(self):
        output = Path("x.json"); output.write_text("ok")
        state = StageState(Path("state.json")); state.save("qa", "key", [output])
        self.assertTrue(state.matches("qa", "key", [output]))
        self.assertFalse(state.matches("qa", "new-key", [output]))
        output.write_text("different")
        self.assertFalse(state.matches("qa", "key", [output]))
        state.save("qa", "key", [output], status="failed")
        self.assertFalse(state.matches("qa", "key", [output]))

    def fake_worker(self, events, *, failed=None, empty=None):
        def run(module, spec, directory, name):
            events.append(module)
            for job in spec["jobs"]:
                source = Path(job["source"])
                source.parent.mkdir(parents=True, exist_ok=True)
                state = StageState(Path(job["state"]))
                if module == "src.workflow_asr":
                    if failed and failed in job["media"]:
                        state.save("asr", job["key"], [], status="failed")
                    elif empty and empty in job["media"]:
                        state.save("asr", job["key"], [], status="empty")
                    else:
                        write_translated_srt(self.src, source)
                        meta = artifact_path(source, ".asr.json"); meta.write_text("[]")
                        state.save("asr", job["key"], [source, meta])
                else:
                    out = Path(job["out"])
                    out.write_text(json.dumps([{"line": b.index, "w": b.text, "n": b.text,
                                               "q": b.text, "grade": "A"} for b in self.src]))
                    state.save("arbitration", job["key"], [out])
            return 0
        return run

    def fake_llm(self, events):
        def call(body, instruction, config, **kwargs):
            events.append(config.stage)
            ids = [int(m) for m in __import__("re").findall(r"^\[(\d+)\]", body, __import__("re").M)]
            if config.stage == "translation":
                return "\n".join(f"[{i}] 你好" for i in ids)
            if config.stage == "source-resolution":
                return "\n".join(f"[{i}] KEEP|Candidates agree" for i in ids)
            return "\n".join(f"[{i}] OK" for i in ids)
        return call

    def args_and_media(self, names=("track01", "track02"), organize=False):
        directory = Path("input/work"); directory.mkdir(parents=True)
        for name in names:
            (directory / f"{name}.wav").write_bytes(b"fake-media-" + name.encode())
        argv = ["pipeline", "--evidence-first", "--asr-strategy", "sequential", "--arbitrate", "--review", "--no-auto-context", "-l", "ja"]
        if organize:
            argv.append("--organize")
        return build_parser().parse_args(argv)

    def test_batch_order_organize_and_resume_without_any_model_calls(self):
        args = self.args_and_media(organize=True)
        events = []
        with patch.object(workflow, "run_worker", side_effect=self.fake_worker(events)), \
             patch.object(quality, "call_llm", side_effect=self.fake_llm(events)):
            self.assertEqual(workflow.run_workflow(args), 0)
            self.assertEqual(events[:2], ["src.workflow_asr", "src.adjudicate"])
            self.assertEqual(events.count("src.adjudicate"), 1)
            self.assertTrue(Path("output/work/final/track01.zh.srt").exists())
            self.assertTrue(Path("output/work/review/track01.workflow.json").exists())
            events.clear()
            self.assertEqual(workflow.run_workflow(args), 0)
            self.assertEqual(events, [])

    def test_force_discards_all_writer_and_delta_approval_caches(self):
        args = self.args_and_media(names=("track01",))
        args.force = True
        paths = [Path("output/work/track01" + suffix) for suffix in
                 (".requests.json", ".requests.repair-delta.json", ".display-requests.json", ".entity-lexicon.json", ".entity-classify.json", ".source-span-requests.json")]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"old-approval": true}')
        def worker(module, spec, directory, name):
            self.assertTrue(all(not path.exists() for path in paths))
            return self.fake_worker([])(module, spec, directory, name)
        with patch.object(workflow, "run_worker", side_effect=worker), \
             patch.object(quality, "call_llm", side_effect=self.fake_llm([])):
            self.assertEqual(workflow.run_workflow(args), 0)
        self.assertFalse(paths[1].exists())

    def test_ensemble_owns_evidence_on_force_and_missing_evidence_resume(self):
        from src.workflow_state import fingerprint
        args = self.args_and_media(names=("track01",))
        args.asr_strategy = "ensemble"
        events = []
        def worker(module, spec, directory, name):
            self.assertEqual(module, "src.ensemble_asr")
            events.append(module)
            for job in spec["jobs"]:
                source = Path(job["source"]); source.parent.mkdir(parents=True, exist_ok=True)
                write_translated_srt(self.src, source)
                meta = Path(job["metadata"]); meta.write_text("{}")
                adj = Path(job["adjudication"])
                adj.write_text(json.dumps([dict(line=b.index,w=b.text,n=b.text,q=b.text,
                                                grade="A",ensemble=True) for b in self.src]))
                state = StageState(Path(job["state"]))
                state.save("asr",job["key"],[source,meta])
                key = fingerprint([quality.VERSION,job["media_hash"],file_hash(source),
                                   spec["source_lang"],spec["batch_size"]])
                state.save("arbitration",key,[adj])
            return 0
        with patch.object(workflow,"run_worker",side_effect=worker), \
             patch.object(quality,"call_llm",side_effect=self.fake_llm(events)):
            self.assertEqual(workflow.run_workflow(args),0)
            events.clear()
            self.assertEqual(workflow.run_workflow(args),0)
            self.assertEqual(events,[])
            args.force = True
            self.assertEqual(workflow.run_workflow(args),0)
            self.assertEqual(events.count("src.ensemble_asr"),1)
            args.force = False; events.clear()
            Path("output/work/track01.adjudication.json").unlink()
            self.assertEqual(workflow.run_workflow(args),0)
            self.assertEqual(events.count("src.ensemble_asr"),1)

    def test_one_failure_does_not_drop_other_tracks_or_organize_failed_unit(self):
        args = self.args_and_media(names=("track01", "track02", "track03"), organize=True)
        events = []
        with patch.object(workflow, "run_worker", side_effect=self.fake_worker(events, failed="track02", empty="track03")), \
             patch.object(quality, "call_llm", side_effect=self.fake_llm(events)):
            self.assertEqual(workflow.run_workflow(args), 1)
        self.assertTrue(Path("output/work/track01.zh.srt").exists())
        self.assertTrue(Path("input/work/track02.wav").exists())
        self.assertFalse(Path("output/work/final").exists())

    def test_external_final_edit_is_preserved_on_resume(self):
        args = self.args_and_media(names=("track01",))
        events = []
        with patch.object(workflow, "run_worker", side_effect=self.fake_worker(events)), \
             patch.object(quality, "call_llm", side_effect=self.fake_llm(events)):
            self.assertEqual(workflow.run_workflow(args), 0)
            path = Path("output/work/track01.zh.srt")
            path.write_text(path.read_text().replace("你好", "人工修改"))
            before = path.read_bytes()
            self.assertEqual(workflow.run_workflow(args), 1)
            self.assertEqual(path.read_bytes(), before)

    def test_per_work_hotwords_do_not_leak(self):
        from src.hotwords import HotwordResult
        from src.work_context import prepare_work_context
        args = self.args_and_media(names=("track01",))
        Path("input/other").mkdir()
        Path("input/other/track01.wav").write_bytes(b"other")
        jobs = workflow.collect_jobs(args)
        # Distinct work titles trigger hotwords even without context files.
        jobs[0].media = jobs[0].media.with_name("去勢手術.wav")
        jobs[1].media = jobs[1].media.with_name("魔法学校.wav")
        seen = []
        def hotwords(**kwargs):
            seen.append(kwargs["title"])
            return HotwordResult([kwargs["title"]], "llm")
        with patch("src.work_context.prepare_work_context", return_value="background"), \
             patch("src.hotwords.generate_hotwords", side_effect=hotwords):
            workflow.prepare_context(jobs, args)
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(jobs[0].hotwords, jobs[1].hotwords)
        self.assertEqual(len(jobs[0].hotwords), 1)


    def test_non_streaming_fallback_works_with_zero_retries(self):
        import requests
        import src.translate as transport
        response = requests.Response(); response.status_code = 400
        error = requests.HTTPError(response=response)
        with patch.object(transport, "_stream_response", side_effect=error), \
             patch.object(transport, "_call_llm_non_streaming", return_value=transport.StreamResult("ok")) as fallback:
            self.assertEqual(transport.call_llm("x", "y", self.cfg), "ok")
            self.assertEqual(fallback.call_count, 1)

    def test_thinking_fallback_overrides_explicit_template_true(self):
        import src.translate as transport
        cfg = replace(self.cfg, retries=1, extra_payload={"chat_template_kwargs": {"enable_thinking": True}})
        with patch.object(transport, "_stream_response", side_effect=[transport.StreamResult(""), transport.StreamResult("ok")]) as stream, \
             patch.object(transport.time, "sleep"):
            self.assertEqual(transport.call_llm("x", "y", cfg, with_thinking=True), "ok")
        final_payload = stream.call_args_list[1].args[1]
        self.assertFalse(final_payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(final_payload["reasoning_effort"], "none")

    def test_stream_usage_chunk_without_choices_is_recorded(self):
        import src.translate as transport
        class Response:
            def raise_for_status(self):
                pass
            def close(self):
                pass
            def iter_lines(self, **kwargs):
                yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
                yield 'data: {"choices":[],"usage":{"completion_tokens":1},"timings":{"cache_n":42}}'
                yield 'data: [DONE]'
        with patch.object(transport.requests, "post", return_value=Response()):
            result = transport._stream_response("unused", {}, 1, 10)
        self.assertEqual(result.usage["completion_tokens"], 1)
        self.assertEqual(result.timings["cache_n"], 42)

    def test_duplicate_even_with_invalid_first_result_is_rejected(self):
        self.assertEqual(quality.parse_records("[1] nonsense\n[1] OK", [1], "qa", self.ev), {})

    def test_scene_prompt_names_exact_target_ids(self):
        text = quality.format_scene([2], self.src, self.ev, self.cfg)
        self.assertIn("本次输出ID必须恰好为: 2", text)
        self.assertIn("禁止输出邻域ID", text)

    def test_different_media_cannot_share_an_organized_output(self):
        args = self.args_and_media(names=("track01",))
        directory = Path("output/work/final"); directory.mkdir(parents=True)
        (directory / "track01.wav").write_bytes(b"different-old-recording")
        with self.assertRaisesRegex(ValueError, "Different media"):
            workflow.collect_jobs(args)

    def test_context_glossary_discards_invented_evidence(self):
        from src.work_context import prepare_work_context
        path = Path("story.txt")
        path.write_text("医師の名前はデニス。\n" + "病院の日常。\n" * 600)
        reply = json.dumps({"summary":"病院", "terms":[
            {"source":"デニス", "translation":"丹尼斯", "evidence":"医師の名前はデニス。"},
            {"source":"魔王", "translation":"魔王", "evidence":"魔王が登場"}]}, ensure_ascii=False)
        with patch("src.work_context.call_llm", return_value=reply):
            context = prepare_work_context([path], [], self.cfg, Path("context.json"))
        self.assertIn("丹尼斯", context)
        self.assertNotIn("魔王", context)


    def test_digital_silence_cannot_become_asr_consensus(self):
        import wave
        from unittest.mock import Mock
        from src.adjudicate import adjudicate
        wav = Path("silence.wav")
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000 * 8)
        source = Path("source.srt"); write_translated_srt(self.src, source)
        n, q = Mock(), Mock()
        out = Path("adj.json")
        adjudicate(source, wav, [1, 2], out, n_model=n, q_model=q)
        n.transcribe.assert_not_called()
        q.transcribe_batch.assert_not_called()
        rows = json.loads(out.read_text())
        self.assertEqual([r["grade"] for r in rows], ["D", "D"])
        self.assertTrue(all("数字静音" in r["note"] for r in rows))


    def test_sparse_issues_pack_together_with_individual_context(self):
        source = blocks(tuple("こんにちは" for _ in range(10)))
        self.assertEqual(quality.scene_groups(source, self.cfg, [1, 5, 10]), [[1, 5, 10]])

    def test_runner_selects_new_default_and_preserves_stage_toggles(self):
        import shutil
        import subprocess
        runner = self.root / "run.sh"
        shutil.copy(self.old_cwd / "run.sh", runner)
        fake = self.root / "fake-python"
        fake.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        env = {**os.environ, "PY": str(fake), "PROOFREAD": "0", "REVIEW": "0",
               "ADJUDICATE": "0", "ORGANIZE": "0", "VOCAB": "unused.txt", "WORKFLOW": "evidence"}
        result = subprocess.run(["bash", str(runner)], env=env, text=True, capture_output=True, check=True)
        args = json.loads(result.stdout)
        self.assertIn("--evidence-first", args)
        self.assertNotIn("--arbitrate", args)
        self.assertNotIn("--review", args)
        self.assertNotIn("--proofread", args)
        self.assertNotIn("--organize", args)
        env["WORKFLOW"] = "legacy"
        result = subprocess.run(["bash", str(runner)], env=env, text=True, capture_output=True, check=True)
        args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
        self.assertNotIn("--evidence-first", args)


    def test_default_auto_context_handles_a_fresh_output_directory(self):
        args = self.args_and_media(names=("track01",))
        args.no_auto_context = False
        events = []
        self.assertFalse(Path("output/work").exists())
        with patch.object(workflow, "run_worker", side_effect=self.fake_worker(events)), \
             patch.object(quality, "call_llm", side_effect=self.fake_llm(events)):
            self.assertEqual(workflow.run_workflow(args), 0)
        self.assertTrue(Path("output/work/track01.zh.srt").exists())


if __name__ == "__main__":
    unittest.main()
