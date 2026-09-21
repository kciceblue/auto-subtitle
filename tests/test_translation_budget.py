"""Offline draft-budget isolation, caching and actual request telemetry checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from main import build_parser, cmd_pipeline
from src import quality, translate, workflow
from src.config import TranslateConfig, TranscribeConfig
from src.evidence import CueEvidence
from src.translate import SrtBlock
from src.workflow_state import read_json


class TranslationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = [SrtBlock(1, '00:00:01,000 --> 00:00:02,000', 'こんにちは。')]
        self.evidence = [CueEvidence(line=1, raw=self.source[0].text, grade='A')]
        self.config = TranslateConfig(max_tokens=2048, retries=2,
            extra_payload={'model':'qwen3.8-27b-dflash','temperature':.3},
            telemetry_path=self.root/'metrics.jsonl')
        self.cache = self.root/'requests.json'

    def test_cli_defaults_and_validation_do_not_touch_transcription_config(self):
        args = build_parser().parse_args(['pipeline'])
        self.assertEqual(args.translation_reasoning_budget, 0)
        self.assertEqual(TranslateConfig().translation_reasoning_budget, 0)
        self.assertEqual(TranscribeConfig().ensemble_source, 'consensus')
        for bad in (-1, 128, 1024, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TranslateConfig(translation_reasoning_budget=bad)
        with self.assertRaises(ValueError):
            TranslateConfig(translation_reasoning_budget=512, max_tokens=512)
        args = build_parser().parse_args(['pipeline','--translation-reasoning-budget','256'])
        with patch.object(workflow,'run_workflow') as run:
            self.assertEqual(cmd_pipeline(args),1)
            run.assert_not_called()

    def test_default_draft_payload_and_retries_are_unchanged(self):
        before = quality.mode(self.config,'translation')
        after = quality.draft_translation_config(self.config)
        self.assertEqual(after,before)
        with patch.object(quality,'call_llm',return_value='[1] 你好。') as llm:
            quality.translate_scenes(self.source,self.evidence,self.config,self.cache)
        self.assertFalse(llm.call_args.kwargs['with_thinking'])
        self.assertEqual(llm.call_args.args[2].retries,2)
        self.assertNotIn('reasoning_budget_tokens',llm.call_args.args[2].extra_payload)

    def test_capped_draft_uses_actual_budget_and_pinned_total_without_mutating_other_stages(self):
        cfg = replace(self.config,translation_reasoning_budget=256)
        original_extra = dict(cfg.extra_payload)
        with patch.object(quality,'call_llm',return_value='[1] 你好。') as llm:
            quality.translate_scenes(self.source,self.evidence,cfg,self.cache)
        actual = llm.call_args.args[2]
        self.assertTrue(llm.call_args.kwargs['with_thinking'])
        self.assertEqual(actual.extra_payload['reasoning_budget_tokens'],256)
        self.assertEqual(actual.extra_payload['max_tokens'],2048)
        self.assertEqual(actual.retries,0)
        self.assertNotIn('reasoning_effort',actual.extra_payload)
        self.assertEqual(cfg.extra_payload,original_extra)
        for stage in ('source-resolution','diagnosis','targeted-repair','verification','entity-extraction','layout'):
            other = quality.mode(cfg,stage)
            self.assertFalse(other.extra_payload['chat_template_kwargs']['enable_thinking'])
            self.assertNotIn('reasoning_budget_tokens',other.extra_payload)
            self.assertEqual(other.retries,2)

    def test_missing_id_retry_keeps_capped_thinking_and_never_replays_valid_ids(self):
        cfg = replace(self.config,translation_reasoning_budget=512)
        with patch.object(quality,'call_llm',side_effect=['invalid','[1] 你好。']) as llm:
            quality.translate_scenes(self.source,self.evidence,cfg,self.cache)
        self.assertEqual(llm.call_count,2)
        for call in llm.call_args_list:
            self.assertTrue(call.kwargs['with_thinking'])
            self.assertEqual(call.args[2].extra_payload['max_tokens'],2048)
            self.assertEqual(call.args[2].extra_payload['reasoning_budget_tokens'],512)

    def test_request_cache_invalidates_budget_but_upstream_config_key_does_not(self):
        with patch.object(quality,'call_llm',return_value='[1] 你好。') as llm:
            for budget in (0,0,256,256,512,0):
                quality.translate_scenes(self.source,self.evidence,
                    replace(self.config,translation_reasoning_budget=budget),self.cache)
        self.assertEqual(llm.call_count,3)
        self.assertEqual(workflow.configuration_key(self.config),
                         workflow.configuration_key(replace(self.config,translation_reasoning_budget=256)))

    def test_changed_budget_reruns_draft_and_downstream_qa_even_for_identical_text(self):
        media = self.root/'media.wav'; media.write_bytes(b'untouched')
        source_path = self.root/'media.srt'; translate.write_translated_srt(self.source,source_path)
        job = workflow.Job(media,source_path,'unit',self.config)
        args = SimpleNamespace(force=False,proofread=False,review=False)
        decisions = [dict(line=1,status='accepted',candidate='W',text=self.source[0].text,reason='raw')]
        with patch.object(workflow,'build_evidence',return_value=self.evidence), \
             patch.object(workflow,'resolve_source',return_value=(self.source,decisions)) as resolve, \
             patch.object(workflow,'translate_scenes',return_value=[replace(self.source[0],text='你好。')]) as draft, \
             patch.object(workflow,'write_quality_result',wraps=quality.write_quality_result) as qa:
            workflow.process_language(job,args)
            workflow.process_language(job,args)
            self.assertEqual((resolve.call_count,draft.call_count,qa.call_count),(1,1,1))
            job.config=replace(self.config,translation_reasoning_budget=256)
            workflow.process_language(job,args)
            self.assertEqual((resolve.call_count,draft.call_count,qa.call_count),(1,2,2))
        record=read_json(job.state_path)['translation']['request_settings']
        self.assertEqual(record['thinking_budget'],256)
        self.assertEqual(record['request_payload']['max_tokens'],2048)

    def test_actual_http_settings_recorded_and_empty_capped_response_does_not_retry_off(self):
        cfg = quality.draft_translation_config(replace(self.config,translation_reasoning_budget=256))
        empty = translate.StreamResult(content='',reasoning_chars=700,finish_reason='length')
        with patch.object(translate,'_stream_response',return_value=empty) as http:
            with self.assertRaises(RuntimeError):
                translate.call_llm('source','instruction',cfg,with_thinking=True)
        self.assertEqual(http.call_count,1)
        payload=http.call_args.args[1]
        self.assertEqual(payload['max_tokens'],2048)
        self.assertEqual(payload['reasoning_budget_tokens'],256)
        metrics=json.loads(cfg.telemetry_path.read_text().splitlines()[0])
        self.assertTrue(metrics['thinking'])
        self.assertEqual(metrics['request_settings']['reasoning_budget_tokens'],256)
        self.assertTrue(metrics['request_settings']['enable_thinking'])
        self.assertNotIn('messages',metrics['request_settings'])


if __name__=='__main__':unittest.main()
