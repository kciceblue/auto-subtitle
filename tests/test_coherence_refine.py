"""Offline local refinement contracts; no model or network calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.coherence import CoherenceError
from src.coherence_refine import refine_coherence
from src.config import TranslateConfig
from src.translate import SrtBlock


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


class CoherenceRefineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / 'refine.json'
        self.source = [SrtBlock(i, f'00:00:0{2*i},000 --> 00:00:0{2*i+1},000', text)
                       for i, text in enumerate(('帰りましょう。', '待って。', 'はい。'), 1)]
        self.draft = [replace(row, text=text) for row, text in zip(self.source, ('回去吧。', '等一等。', '好。'))]
        self.metadata = {'windows': [{'start': 0., 'end': 3.5}, {'start': 3.5, 'end': 5.5}, {'start': 5.5, 'end': 8.}],
                         'raw_transcripts': [{'q': '第一条原始识别', 'w': '第一条另一候选'},
                                             {'q': '第二条原始识别', 'w': ''},
                                             {'q': '第三条原始识别', 'w': '第三条另一候选'}],
                         'uncertain': [1, 2, 3], 'review_notes': 'EXTERNAL NOTES MUST NOT ENTER PROMPTS'}
        self.context = self.root / 'context.txt'
        self.context.write_text('完整的原始作品背景。', encoding='utf-8')
        self.config = TranslateConfig(endpoint='http://127.0.0.1:18101/v1/chat/completions',
            max_tokens=4096, retries=5, context_files=[self.context],
            extra_payload={'model': 'local-writer', 'temperature': .7, 'seed': 123,
                           'chat_template_kwargs': {'enable_thinking': True}})

    def invoke(self, answers, *, mode='edit-evidence', **kwargs):
        with patch('src.coherence_refine.call_llm', side_effect=answers) as call:
            final, ledger = refine_coherence(self.source, self.draft, self.metadata, self.config,
                                             self.cache, mode=mode, batch_size=2, **kwargs)
        return final, ledger, call

    def saved(self):
        return json.loads(self.cache.read_text(encoding='utf-8'))

    def entry(self, saved):
        return next(iter(next(iter(saved['runs'].values()))['batches'].values()))

    def edits(self):
        return [encoded({'1': '回去吧。', '2': '请等一等。'}), encoded({'1': '好。'})]

    def test_evidence_edit_preserves_inputs_geometry_and_frozen_context(self):
        self.config.line_notes = {1: ['EXTERNAL LINE FINDING']}
        before = deepcopy((self.source, self.draft, self.metadata, self.config))
        final, ledger, call = self.invoke(self.edits())
        self.assertEqual((self.source, self.draft, self.metadata, self.config), before)
        self.assertEqual(final[1].text, '请等一等。')
        for args in call.call_args_list:
            self.assertIn('完整的原始作品背景。', args.args[1])
            self.assertIn('回去吧。\n等一等。\n好。', args.args[1])
            self.assertNotIn('请等一等。', args.args[1])
            self.assertNotIn('EXTERNAL', args.args[0] + args.args[1])
            self.assertEqual(args.args[2].extra_payload['seed'], 123)
            self.assertEqual(args.args[2].retries, 0)
            self.assertFalse(args.kwargs['with_thinking'])
        first, last = call.call_args_list
        self.assertEqual(first.args[1], last.args[1])
        self.assertNotIn('第一条原始识别', first.args[1])
        first_data = json.loads(first.args[0])
        self.assertEqual(set(first_data), {'utterances', 'raw_asr_evidence'})
        self.assertEqual(first_data['raw_asr_evidence'][0]['recognition'], self.metadata['raw_transcripts'][0])
        self.assertEqual(set(first_data['utterances']), {'1', '2'})
        self.assertIn('第一条原始识别', first.args[0])
        self.assertIn('第二条原始识别', first.args[0])
        self.assertNotIn('第三条原始识别', first.args[0])
        self.assertIn('第三条原始识别', last.args[0])
        self.assertNotIn('第一条原始识别', last.args[0])
        self.assertEqual([(r.index, r.ts_line) for r in final], [(r.index, r.ts_line) for r in self.source])
        self.assertTrue(all(r['status'] == 'UNVERIFIED' and not r['source_uncertainty_cleared'] for r in ledger))
        self.assertEqual(next(iter(self.saved()['runs'].values()))['binding']['metadata'], self.metadata)

    def test_critic_is_chinese_only_and_edits_only_locally_flagged_ids(self):
        answers = [encoded({'issues': [{'id': 2, 'reason': '这句中文搭配显得生硬。'}]}), encoded({'1': '请等一下。'})]
        final, ledger, call = self.invoke(answers, mode='critic-edit')
        self.assertEqual(call.call_count, 2)
        critic, edit = call.call_args_list
        self.assertEqual(json.loads(critic.args[0]), {'1': '回去吧。', '2': '等一等。', '3': '好。'})
        self.assertNotIn('第一条原始识别', critic.args[1])
        self.assertNotIn('完整的原始作品背景。', critic.args[1])
        self.assertTrue(critic.kwargs['with_thinking'])
        self.assertEqual(critic.args[2].extra_payload['reasoning_budget_tokens'], 1024)
        self.assertNotIn('reasoning_effort', critic.args[2].extra_payload)
        rows = json.loads(edit.args[0])['utterances']
        self.assertEqual(set(rows), {'1'})
        self.assertEqual(rows['1']['source'], '待って。')
        self.assertEqual(rows['1']['local_diagnosis'], '这句中文搭配显得生硬。')
        self.assertIn('第二条原始识别', edit.args[0])
        self.assertNotIn('第一条原始识别', edit.args[0])
        self.assertEqual([r.text for r in final], ['回去吧。', '请等一下。', '好。'])
        self.assertEqual([r['edited'] for r in ledger], [False, True, False])
        self.assertEqual([r['local_diagnosis']['status'] for r in ledger], ['NOT_FLAGGED', 'ISSUE', 'NOT_FLAGGED'])

    def test_target_editor_never_receives_source_or_asr_and_bounds_reasoning(self):
        final, ledger, call = self.invoke(self.edits(), mode='target-edit', edit_budget=512)
        for request in call.call_args_list:
            body, instruction, cfg = request.args
            self.assertTrue(all(set(row) == {'chinese'} for row in json.loads(body).values()))
            self.assertNotIn('原始识别', instruction)
            self.assertNotIn('帰りましょう', body + instruction)
            self.assertTrue(request.kwargs['with_thinking'])
            self.assertEqual(cfg.extra_payload['reasoning_budget_tokens'], 512)
        self.assertTrue(all(not row['source_exposed_to_editor'] for row in ledger))
        self.assertEqual(final[1].text, '请等一等。')

    def test_target_critic_passes_only_local_reasons_and_numeric_external_score(self):
        answers = [encoded({'issues': [{'id': 2, 'reason': '本地诊断指出中文搭配问题。'}]}),
                   encoded({'1': '请等一下。'})]
        final, ledger, call = self.invoke(answers, mode='target-critic-edit', prior_score=3)
        critic, edit = call.call_args_list
        self.assertIn('整体评分是3', critic.args[1])
        self.assertEqual(json.loads(edit.args[0]), {'1': {'chinese': '等一等。',
                                                       'local_diagnosis': '本地诊断指出中文搭配问题。'}})
        self.assertNotIn('原始识别', edit.args[1])
        self.assertNotIn('EXTERNAL', edit.args[1])
        self.assertEqual(ledger[1]['external_feedback_scope'], 'score_only')
        self.assertFalse(ledger[1]['source_uncertainty_cleared'])
        self.assertEqual(final[1].text, '请等一下。')
        with self.assertRaises(ValueError):
            self.invoke([], mode='target-edit', prior_score={'findings': 'disallowed'})

    def test_scene_critic_batches_keep_global_ids_timestamps_and_complete_numbered_context(self):
        before = deepcopy((self.source, self.draft, self.metadata, self.config))
        answers = [encoded({'issues': [{'id': 2, 'reason': '这句中文搭配显得生硬。'}]}),
                   encoded({'issues': []}), encoded({'1': '请等一下。'})]
        final, ledger, call = self.invoke(answers, mode='scene-critic-edit')
        self.assertEqual((self.source, self.draft, self.metadata, self.config), before)
        self.assertEqual([item.args[2].stage for item in call.call_args_list],
                         ['local_coherence_scene_diagnosis', 'local_coherence_scene_diagnosis',
                          'local_coherence_refine'])
        first, second, edit = call.call_args_list
        self.assertEqual(first.args[1], second.args[1])
        whole = json.loads(first.args[1].split('全片中文只读上下文（全局编号）：\n', 1)[1])
        self.assertEqual(whole, {str(row.index): row.text for row in self.draft})
        for request, rows in ((first, self.draft[:2]), (second, self.draft[2:])):
            body = json.loads(request.args[0])
            self.assertEqual(body, {str(row.index): {'global_id': row.index,
                'timestamp': row.ts_line, 'chinese': row.text} for row in rows})
            schema = request.args[2].extra_payload['response_format']['schema']
            self.assertEqual(schema['properties']['issues']['items']['properties']['id']['enum'],
                             [row.index for row in rows])
            self.assertTrue(request.kwargs['with_thinking'])
            self.assertNotIn('原始识别', request.args[0] + request.args[1])
            self.assertNotIn('帰りましょう', request.args[0] + request.args[1])
            self.assertNotIn('EXTERNAL', request.args[0] + request.args[1])
        edited = json.loads(edit.args[0])
        self.assertEqual(set(edited['utterances']), {'1'})
        self.assertEqual(edited['utterances']['1']['source'], self.source[1].text)
        self.assertEqual(edited['raw_asr_evidence'][0]['recognition'], self.metadata['raw_transcripts'][1])
        self.assertEqual([row.text for row in final], ['回去吧。', '请等一下。', '好。'])
        self.assertEqual([(row.index, row.ts_line) for row in final],
                         [(row.index, row.ts_line) for row in self.source])
        self.assertEqual([row['edited'] for row in ledger], [False, True, False])
        self.assertTrue(all(row['status'] == 'UNVERIFIED' and not row['source_uncertainty_cleared']
                            for row in ledger))

    def test_scene_critic_last_batch_global_id_maps_to_local_editor_id(self):
        answers = [encoded({'issues': []}),
                   encoded({'issues': [{'id': 3, 'reason': '本地诊断指出中文搭配问题。'}]}),
                   encoded({'1': self.draft[2].text})]
        final, ledger, call = self.invoke(answers, mode='scene-critic-edit')
        self.assertEqual(final, self.draft)
        editor = json.loads(call.call_args_list[-1].args[0])
        self.assertEqual(set(editor['utterances']), {'1'})
        self.assertEqual(editor['utterances']['1']['source'], self.source[2].text)
        self.assertEqual(editor['raw_asr_evidence'][0]['recognition'], self.metadata['raw_transcripts'][2])
        self.assertEqual([row['edited'] for row in ledger], [False, False, True])
        self.assertNotEqual(ledger[0]['diagnosis_request_key'], ledger[2]['diagnosis_request_key'])

    def test_scene_critic_cannot_report_read_only_ids_from_another_batch(self):
        bad = encoded({'issues': [{'id': 1, 'reason': '这条不属于当前批次。'}]})
        before = deepcopy((self.source, self.draft, self.metadata))
        with patch('src.coherence_refine.call_llm', side_effect=[encoded({'issues': []}), bad, bad]) as call:
            with self.assertRaises(CoherenceError):
                refine_coherence(self.source, self.draft, self.metadata, self.config, self.cache,
                                 mode='scene-critic-edit', batch_size=2)
        self.assertEqual(call.call_count, 3)
        self.assertTrue(all(item.args[2].stage == 'local_coherence_scene_diagnosis'
                            for item in call.call_args_list))
        self.assertEqual((self.source, self.draft, self.metadata), before)
        run = next(iter(self.saved()['runs'].values()))
        last = next(entry for entry in run['batches'].values() if entry['request']['global_ids'] == [3])
        self.assertEqual([attempt['response'] for attempt in last['attempts']], [bad, bad])

    def test_scene_cache_binds_prompt_and_reparses_global_ids_per_batch(self):
        from src.coherence_refine import SCENE_CRITIC, VERSION
        answers = [encoded({'issues': []}), encoded({'issues': []})]
        first, _, _ = self.invoke(answers, mode='scene-critic-edit')
        second, ledger, call = self.invoke([], mode='scene-critic-edit')
        self.assertEqual(first, second)
        call.assert_not_called()
        self.assertTrue(all(row['cached'] for row in ledger))
        saved = self.saved()
        run = next(iter(saved['runs'].values()))
        self.assertEqual(run['binding']['version'], VERSION)
        self.assertEqual(run['binding']['prompts']['scene_critic'], SCENE_CRITIC)
        self.assertEqual(run['binding']['metadata'], self.metadata)
        last = next(entry for entry in run['batches'].values() if entry['request']['global_ids'] == [3])
        bad = encoded({'issues': [{'id': 1, 'reason': '不属于当前批次。'}]})
        last.update(response=bad, response_sha256=hashlib.sha256(bad.encode()).hexdigest())
        self.cache.write_text(json.dumps(saved), encoding='utf-8')
        final, ledger, call = self.invoke([encoded({'issues': []})], mode='scene-critic-edit')
        self.assertEqual(final, self.draft)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(set(json.loads(call.call_args.args[0])), {'3'})
        self.assertEqual([row['cached'] for row in ledger], [True, True, False])

    def test_scan_checks_every_cue_before_repairs_and_uses_frozen_context(self):
        answers = [encoded({'1': {'status': 'OK', 'reason': ''}, '2': {'status': 'ISSUE', 'reason': '中文表达不自然。'}}),
                   encoded({'1': {'status': 'OK', 'reason': ''}}), encoded({'1': '请等一下。'})]
        final, ledger, call = self.invoke(answers, mode='scan-edit')
        self.assertEqual([a.args[2].stage for a in call.call_args_list],
                         ['local_coherence_scan', 'local_coherence_scan', 'local_coherence_refine'])
        for args in call.call_args_list:
            self.assertIn('回去吧。\n等一等。\n好。', args.args[1])
        self.assertEqual(json.loads(call.call_args_list[1].args[0]), {'1': '好。'})
        self.assertEqual(final[1].text, '请等一下。')
        self.assertEqual([r['local_diagnosis']['status'] for r in ledger], ['OK', 'ISSUE', 'OK'])
        self.assertTrue(all(r['status'] == 'UNVERIFIED' for r in ledger))

    def test_no_issues_returns_an_unchanged_unverified_copy_without_edit_calls(self):
        final, ledger, call = self.invoke([encoded({'issues': []})], mode='critic-edit')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(final, self.draft)
        self.assertIsNot(final[0], self.draft[0])
        self.assertTrue(all(not r['edited'] and r['status'] == 'UNVERIFIED' for r in ledger))

    def test_zero_budget_disables_diagnosis_thinking(self):
        _, _, call = self.invoke([encoded({'issues': []})], mode='critic-edit', critic_budget=0)
        self.assertFalse(call.call_args.kwargs['with_thinking'])
        self.assertNotIn('reasoning_budget_tokens', call.call_args.args[2].extra_payload)
        self.assertEqual(call.call_args.args[2].extra_payload['reasoning_effort'], 'none')

    def test_good_cache_reparses_diagnosis_and_editor_responses(self):
        answers = [encoded({'issues': [{'id': 2, 'reason': '中文表达不自然。'}]}), encoded({'1': '请等一下。'})]
        first, _, _ = self.invoke(answers, mode='critic-edit')
        second, ledger, call = self.invoke([], mode='critic-edit')
        self.assertEqual(first, second)
        call.assert_not_called()
        self.assertTrue(all(r['cached'] for r in ledger))

    def test_tampered_critic_is_reparsed_even_with_matching_hash(self):
        self.invoke([encoded({'issues': []})], mode='critic-edit')
        saved = self.saved()
        item = self.entry(saved)
        bad = '{"issues":[{"id":2,"reason":"x"},{"id":2,"reason":"y"}]}'
        item.update(response=bad, response_sha256=hashlib.sha256(bad.encode()).hexdigest())
        self.cache.write_text(json.dumps(saved), encoding='utf-8')
        final, _, call = self.invoke([encoded({'issues': []})], mode='critic-edit')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(final, self.draft)
        self.assertTrue(self.entry(self.saved())['cache_rejections'])

    def test_raw_metadata_changes_invalidate_cached_edits(self):
        self.invoke(self.edits())
        self.metadata['raw_transcripts'][0]['q'] = '新原始候选'
        _, ledger, call = self.invoke(self.edits())
        self.assertEqual(call.call_count, 2)
        self.assertEqual(len(self.saved()['runs']), 2)
        self.assertTrue(all(not r['cached'] for r in ledger))

    def test_malformed_critic_records_fail_after_two_attempts(self):
        invalid = ['{}', '{"issues":[],"extra":0}', '{"issues":[{"id":true,"reason":"x"}]}',
                   '{"issues":[{"id":4,"reason":"x"}]}', '{"issues":[{"id":2,"reason":""}]}',
                   '{"issues":[{"id":2,"reason":"x","correction":"不该有"}]}',
                   '{"issues":[{"id":2,"reason":"x"},{"id":2,"reason":"x"}]}']
        for i, bad in enumerate(invalid):
            with self.subTest(bad=bad), patch('src.coherence_refine.call_llm', return_value=bad) as call:
                with self.assertRaises(CoherenceError):
                    refine_coherence(self.source, self.draft, self.metadata, self.config,
                                     self.root / f'critic-{i}.json', mode='critic-edit', batch_size=2)
            self.assertEqual(call.call_count, 2)

    def test_incomplete_or_invalid_scan_cannot_skip_a_cue(self):
        invalid = [encoded({'1': {'status': 'OK', 'reason': ''}}),
                   encoded({'1': {'status': 'OK', 'reason': '有问题却标OK'}, '2': {'status': 'OK', 'reason': ''}}),
                   encoded({'1': {'status': 'ISSUE', 'reason': ''}, '2': {'status': 'OK', 'reason': ''}})]
        for i, bad in enumerate(invalid):
            with self.subTest(bad=bad), patch('src.coherence_refine.call_llm', return_value=bad) as call:
                with self.assertRaises(CoherenceError):
                    refine_coherence(self.source, self.draft, self.metadata, self.config,
                                     self.root / f'scan-{i}.json', mode='scan-edit', batch_size=2)
            self.assertEqual(call.call_count, 2)

    def test_later_edit_failure_preserves_original_objects_and_output_files(self):
        output = self.root / 'existing.zh.srt'
        output.write_bytes(b'untouched output')
        self.config.output_srt = output
        before = deepcopy((self.source, self.draft, self.metadata))
        answers = [encoded({'1': '回家吧。', '2': '请等一下。'}), '{}', '{}']
        with patch('src.coherence_refine.call_llm', side_effect=answers):
            with self.assertRaises(CoherenceError):
                refine_coherence(self.source, self.draft, self.metadata, self.config, self.cache, batch_size=2)
        self.assertEqual((self.source, self.draft, self.metadata), before)
        self.assertEqual(output.read_bytes(), b'untouched output')
        self.assertEqual(len(next(iter(self.saved()['runs'].values()))['batches']), 2)

    def test_bad_metadata_and_alignment_fail_before_requests(self):
        variants = [dict(self.metadata, raw_transcripts=self.metadata['raw_transcripts'][:-1]),
                    dict(self.metadata, windows=[]),
                    dict(self.metadata, raw_transcripts=[{'q': {'review': 'not raw text'}}, *self.metadata['raw_transcripts'][1:]])]
        bad = deepcopy(self.metadata)
        bad['windows'][0]['end'] = float('nan')
        variants.append(bad)
        bad = deepcopy(self.metadata)
        bad['windows'][2].update(start=9, end=10)
        variants.append(bad)
        for metadata in variants:
            with self.subTest(metadata=metadata), patch('src.coherence_refine.call_llm') as call:
                with self.assertRaises(ValueError):
                    refine_coherence(self.source, self.draft, metadata, self.config, self.cache)
                call.assert_not_called()
        with patch('src.coherence_refine.call_llm') as call:
            with self.assertRaises(ValueError):
                refine_coherence(self.source, self.draft[:-1], self.metadata, self.config, self.cache)
            call.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_nonlocal_writer_and_unbounded_budget_fail_before_diagnosis(self):
        configs = [replace(self.config, endpoint='https://outside.invalid/v1/chat/completions'),
                   replace(self.config, extra_payload={'model': 'gpt-6-astra'})]
        with patch('src.coherence_refine.call_llm') as call:
            for cfg in configs:
                with self.assertRaises(ValueError):
                    refine_coherence(self.source, self.draft, self.metadata, cfg, self.cache, mode='critic-edit')
            with self.assertRaises(ValueError):
                refine_coherence(self.source, self.draft, self.metadata, self.config, self.cache,
                                 mode='critic-edit', critic_budget=4096)
            call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
