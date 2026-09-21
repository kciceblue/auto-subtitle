"""Offline hotword provenance and cache-isolation regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
from src import hotwords, workflow
from src.config import TranscribeConfig


class HotwordModeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.addCleanup(self.temporary.cleanup)
        self.vocab = self.root / "vocab.txt"
        self.words = self.root / "hotwords.txt"
        self.context = self.root / "context.txt"
        self.context.write_text("正しい用語は七影蝶。七栄町や失憶帳ではない。", encoding="utf-8")
        self.cache = self.root / "hotword-cache.json"

    def test_explicit_combines_files_rhs_only_deduplicates_and_never_reads_context(self):
        self.vocab.write_text("\ufeff# comment\n七栄町 -> 七影蝶\n七影蝶\nBDSM\n捨てる->\n", encoding="utf-8")
        self.words.write_text("# another comment\nbdsm\n失憶帳->途中->記憶\n稲荷\n", encoding="utf-8")
        with patch.object(hotwords, "_call_llm") as model, patch.object(hotwords, "_read_context") as read_context:
            result = hotwords.generate_hotwords(context_files=[self.context], vocab_file=self.vocab,
                                               hotwords_file=self.words, title="悪い候補",
                                               cache_file=self.cache, mode="explicit")
        self.assertEqual(result.hotwords, ["七影蝶", "BDSM", "記憶", "稲荷"])
        self.assertEqual(result.source, "explicit")
        model.assert_not_called()
        read_context.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_explicit_bypasses_poisoned_model_cache_without_overwriting_it(self):
        self.vocab.write_text("七影蝶\n", encoding="utf-8")
        with patch.object(hotwords, "_call_llm", return_value=["七栄町", "失憶帳"]):
            generated = hotwords.generate_hotwords([self.context], self.vocab, cache_file=self.cache)
        before = self.cache.read_bytes()
        with patch.object(hotwords, "_call_llm") as model:
            explicit = hotwords.generate_hotwords([self.context], self.vocab, cache_file=self.cache, mode="explicit")
            old = hotwords.generate_hotwords([self.context], self.vocab, cache_file=self.cache)
        model.assert_not_called()
        self.assertEqual(explicit.hotwords, ["七影蝶"])
        self.assertNotEqual(explicit.cache_key, generated.cache_key)
        self.assertEqual(old.hotwords, generated.hotwords)
        self.assertEqual(old.source, "cache")
        self.assertEqual(self.cache.read_bytes(), before)
        self.cache.write_text("not JSON", encoding="utf-8")
        self.assertEqual(hotwords.generate_hotwords(vocab_file=self.vocab, cache_file=self.cache,
                                                  mode="explicit").hotwords, ["七影蝶"])
        self.assertEqual(self.cache.read_text(), "not JSON")

    def test_explicit_recomputes_changed_terms_independently_of_context_and_endpoint(self):
        self.vocab.write_text("蒼\n", encoding="utf-8")
        first = hotwords.generate_hotwords(vocab_file=self.vocab, mode="explicit")
        changed_context = hotwords.generate_hotwords(vocab_file=self.vocab, mode="explicit",
                                                     context_files=[self.context], title="違うタイトル",
                                                     endpoint="http://unused", extra_payload={"model": "unused"})
        self.assertEqual(first, changed_context)
        self.vocab.write_text("藍\n", encoding="utf-8")
        changed_vocab = hotwords.generate_hotwords(vocab_file=self.vocab, mode="explicit")
        self.assertNotEqual(first.cache_key, changed_vocab.cache_key)
        self.assertEqual(changed_vocab.hotwords, ["藍"])

    def test_explicit_requires_nonempty_supplied_terms(self):
        for text in ("", "# comments only\n", "誤り -> \n"):
            self.vocab.write_text(text, encoding="utf-8")
            with self.subTest(text=text), patch.object(hotwords, "_call_llm") as model:
                with self.assertRaisesRegex(ValueError, "requires usable terms"):
                    hotwords.generate_hotwords(context_files=[self.context], vocab_file=self.vocab,
                                              mode="explicit")
                model.assert_not_called()
        with self.assertRaisesRegex(ValueError, "requires usable terms"):
            hotwords.generate_hotwords(context_files=[self.context], mode="explicit")
        with self.assertRaises(FileNotFoundError):
            hotwords.generate_hotwords(vocab_file=self.root / "missing.txt", mode="explicit")

    def test_explicit_cap_preserves_first_occurrence_order(self):
        self.vocab.write_text("\n".join(f"名称{i}" for i in range(45)), encoding="utf-8")
        result = hotwords.generate_hotwords(vocab_file=self.vocab, mode="explicit")
        self.assertEqual(result.hotwords, [f"名称{i}" for i in range(40)])

    def test_none_uses_no_terms_context_extraction_or_generated_cache(self):
        self.vocab.write_text("七栄町->七影蝶\n", encoding="utf-8")
        self.cache.write_text('{"poisoned":["七栄町"]}', encoding="utf-8")
        before = self.cache.read_bytes()
        with patch.object(hotwords, "_call_llm") as model, patch.object(hotwords, "_read_context") as context:
            result = hotwords.generate_hotwords(context_files=[self.context], vocab_file=self.vocab,
                                               hotwords_file=self.root / "unused-file.txt",
                                               cache_file=self.cache, mode="none")
        self.assertEqual(result.hotwords, [])
        self.assertEqual(result.source, "disabled")
        model.assert_not_called()
        context.assert_not_called()
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual(TranscribeConfig(hotword_mode="none").hotword_mode, "none")

    def test_none_workflow_keeps_vocabulary_and_context_for_translation(self):
        self.vocab.write_text("七栄町->七影蝶\n", encoding="utf-8")
        self.words.write_text("学校\n", encoding="utf-8")
        args = self.workflow_args()
        args.hotword_mode = "none"
        args.context = [self.context]
        jobs = workflow.collect_jobs(args)
        item = SimpleNamespace(path=self.context, kind="synopsis")
        with patch("src.context_scan.classify_explicit", return_value={str(self.context): item}), \
             patch("src.work_context.prepare_work_context", return_value="translation background"), \
             patch.object(hotwords, "_call_llm") as model:
            workflow.prepare_context(jobs, args)
        self.assertEqual(jobs[0].hotwords, [])
        self.assertEqual(jobs[0].config.vocab_file, self.vocab)
        self.assertEqual(jobs[0].config.context_summary, "translation background")
        self.assertIn(self.context, jobs[0].config.context_files)
        model.assert_not_called()

    def test_none_transcribe_passes_empty_asr_bias_and_skips_context_models(self):
        self.vocab.write_text("学校\n", encoding="utf-8")
        media = self.root / "input.wav"
        media.write_bytes(b"placeholder")
        args = main.build_parser().parse_args(["transcribe", str(media), "--hotword-mode", "none",
                                              "--vocab", str(self.vocab), "--context", str(self.context)])
        with patch("transcribe.run_transcribe", return_value=(0, [])) as asr, \
             patch("src.context_scan.classify_explicit") as classify, \
             patch.object(hotwords, "_call_llm") as model:
            self.assertEqual(main.cmd_transcribe(args), 0)
        self.assertEqual(asr.call_args.args[0].hotwords, [])
        self.assertEqual(asr.call_args.args[0].hotword_mode, "none")
        classify.assert_not_called()
        model.assert_not_called()

    def test_modes_are_validated_and_model_mode_is_default(self):
        self.assertEqual(TranscribeConfig().hotword_mode, "model")
        self.assertEqual(main.build_parser().parse_args(["pipeline"]).hotword_mode, "model")
        self.assertEqual(main.build_parser().parse_args(["transcribe", "--hotword-mode", "explicit"]).hotword_mode, "explicit")
        with self.assertRaises(ValueError):
            TranscribeConfig(hotword_mode="typo")
        with self.assertRaises(ValueError):
            hotwords.generate_hotwords(mode="typo")

    def test_transcribe_explicit_avoids_context_models_and_uses_both_files(self):
        self.vocab.write_text("七栄町->七影蝶\n", encoding="utf-8")
        self.words.write_text("BDSM\n", encoding="utf-8")
        media = self.root / "input.wav"
        media.write_bytes(b"placeholder")
        args = main.build_parser().parse_args(["transcribe", str(media), "--hotword-mode", "explicit",
                                              "--vocab", str(self.vocab), "--hotwords-file", str(self.words),
                                              "--context", str(self.context)])
        with patch("transcribe.run_transcribe", return_value=(0, [])) as asr, \
             patch("src.context_scan.classify_explicit") as classify, \
             patch("src.context_scan.scan_context_files") as scan, \
             patch.object(hotwords, "_call_llm") as model:
            self.assertEqual(main.cmd_transcribe(args), 0)
        self.assertEqual(asr.call_args.args[0].hotwords, ["七影蝶", "BDSM"])
        self.assertEqual(asr.call_args.args[0].hotword_mode, "explicit")
        classify.assert_not_called()
        scan.assert_not_called()
        model.assert_not_called()

    def test_explicit_empty_cli_fails_before_any_model_or_asr(self):
        for command in ("transcribe", "pipeline"):
            args = main.build_parser().parse_args([command, "--hotword-mode", "explicit"])
            with self.subTest(command=command), patch("transcribe.run_transcribe") as asr, \
                 patch("src.context_scan.scan_context_files") as scan, \
                 patch.object(hotwords, "_call_llm") as model:
                self.assertEqual(getattr(main, "cmd_" + command)(args), 1)
                asr.assert_not_called()
                scan.assert_not_called()
                model.assert_not_called()

    def workflow_args(self):
        media = self.root / "input/work/track01.wav"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"placeholder")
        return main.build_parser().parse_args(["pipeline", "--evidence-first", "--no-auto-context",
                                              "--vocab", str(self.vocab), "--hotwords-file", str(self.words),
                                              "--hotword-mode", "explicit", "-l", "ja"])

    def test_workflow_keeps_translation_context_with_explicit_asr_terms(self):
        self.vocab.write_text("七栄町->七影蝶\n", encoding="utf-8")
        self.words.write_text("BDSM\n", encoding="utf-8")
        args = self.workflow_args()
        args.context = [self.context]
        jobs = workflow.collect_jobs(args)
        item = SimpleNamespace(path=self.context, kind="synopsis")
        with patch("src.context_scan.classify_explicit", return_value={str(self.context): item}), \
             patch("src.work_context.prepare_work_context", return_value="translation background") as context, \
             patch.object(hotwords, "_call_llm") as model:
            workflow.prepare_context(jobs, args)
        self.assertEqual(jobs[0].hotwords, ["七影蝶", "BDSM"])
        self.assertEqual(jobs[0].config.context_summary, "translation background")
        self.assertEqual(context.call_args.args[0], [self.context])
        model.assert_not_called()

    def test_workflow_explicit_empty_fails_before_context_preparation(self):
        self.vocab.write_text("# blank\n", encoding="utf-8")
        self.words.write_text("", encoding="utf-8")
        args = self.workflow_args()
        jobs = workflow.collect_jobs(args)
        with patch("src.work_context.prepare_work_context") as context, patch.object(hotwords, "_call_llm") as model:
            with self.assertRaises(ValueError):
                workflow.prepare_context(jobs, args)
        context.assert_not_called()
        model.assert_not_called()

    def test_hotword_mode_changes_asr_fingerprint_even_when_terms_match(self):
        self.vocab.write_text("七影蝶\n", encoding="utf-8")
        self.words.write_text("七影蝶\n", encoding="utf-8")
        args = self.workflow_args()
        specs = []
        def prepare(jobs, args):
            for job in jobs:
                job.hotwords = ["七影蝶"]
        def worker(module, spec, directory, name):
            specs.append(spec)
            return 1  # No output: stop before language processing or model calls.
        with patch.object(workflow, "prepare_context", side_effect=prepare), \
             patch.object(workflow, "run_worker", side_effect=worker):
            self.assertEqual(workflow.run_workflow(args), 1)
            args.hotword_mode = "model"
            self.assertEqual(workflow.run_workflow(args), 1)
            args.hotword_mode = "none"
            self.assertEqual(workflow.run_workflow(args), 1)
        self.assertEqual([s["config"]["hotword_mode"] for s in specs], ["explicit", "model", "none"])
        self.assertEqual(len({s["jobs"][0]["key"] for s in specs}), 3)
        self.assertEqual(specs[0]["jobs"][0]["hotwords"], specs[1]["jobs"][0]["hotwords"])


if __name__ == "__main__":
    unittest.main()
