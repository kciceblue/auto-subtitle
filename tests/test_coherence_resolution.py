"""Synthetic repair-resolution contracts; all local HTTP and generation mocked."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.coherence import CoherenceError, _text_hash
from src.config import TranslateConfig
from src.translate import SrtBlock
from src.workflow_state import fingerprint

from src import coherence_resolution as R


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def diagnosis(*states):
    return encoded({str(i): {'status': status, 'reason': 'local issue' if status == 'ISSUE' else '',
                            'context_ids': [2] if status == 'ISSUE' else []}
                    for i, status in enumerate(states, 1)})


def resolution(*states):
    return encoded({str(i): {'status': status, 'reason': 'local check'} for i, status in enumerate(states, 1)})


class ResolutionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cache = self.directory / 'requests.json'
        self.source = [SrtBlock(1, '00:00:00,000 --> 00:00:02,000', 'SOURCE_SECRET_A'),
                       SrtBlock(2, '00:00:02,000 --> 00:00:04,000', 'SOURCE_SECRET_B')]
        self.draft = [replace(row, text=text) for row, text in zip(self.source, ('甲句。', '乙句。'))]
        self.config = TranslateConfig(endpoint='http://127.0.0.1:18101/v1/chat/completions',
            max_tokens=2048, timeout=30, retries=0,
            extra_payload={'model': 'synthetic-local-writer', 'temperature': .7, 'seed': 123})
        self.preflight = patch.object(R, 'capacity_check', return_value={'fits': True, 'prompt_tokens': 100,
                                                                       'full_context_preserved': True})
        self.capacity = self.preflight.start()
        self.addCleanup(self.preflight.stop)

    def invoke(self, answers, **kwargs):
        with patch('src.coherence_refine.call_llm', side_effect=answers) as call:
            result = R.refine_resolution(self.source, self.draft, self.config, self.cache,
                                        critic_budget=128, edit_budget=128, **kwargs)
        return *result, call

    def saved(self):
        return json.loads(self.cache.read_text(encoding='utf-8'))

    def test_persistence_retries_unchanged_and_stops_when_resolved(self):
        final, ledger, call = self.invoke([diagnosis('ISSUE', 'OK'), encoded({'1': '甲句。'}),
            resolution('persists'), encoded({'1': '修订甲句。'}), resolution('resolved')])
        self.assertEqual(len(call.call_args_list), 5)
        self.assertEqual(final[0].text, '修订甲句。')
        self.assertEqual(ledger[0]['repair_attempts'], 2)
        self.assertEqual([r['status'] for r in ledger[0]['resolution_history']], ['persists', 'resolved'])
        self.assertFalse(ledger[0]['unresolved'])
        self.assertEqual(ledger[1]['resolution_status'], 'not_flagged')
        repair_calls = [c for c in call.call_args_list if c.args[2].stage.endswith('_repair')]
        self.assertEqual([c.args[2].extra_payload['seed'] for c in repair_calls], [124, 125])
        retry = json.loads(repair_calls[1].args[0])['items']['1']
        self.assertEqual(retry['previous_resolution']['status'], 'persists')
        self.assertEqual(retry['global_id'], 1)
        self.assertEqual(retry['timestamp'], self.draft[0].ts_line)
        self.assertEqual(ledger[0]['model'], 'synthetic-local-writer')
        self.assertIn('settings', ledger[0]['resolution_history'][-1]['resolution_receipt'])
        self.assertTrue(all(r['status'] == 'UNVERIFIED' and not r['source_verified'] for r in ledger))

    def test_three_failed_repairs_leave_explicit_final_hash_bound_unresolved(self):
        answers = [diagnosis('ISSUE', 'OK')]
        for _ in range(3):
            answers += [encoded({'1': '甲句。'}), resolution('persists')]
        final, ledger, call = self.invoke(answers)
        self.assertEqual(final, self.draft)
        self.assertEqual(len(call.call_args_list), 7)
        self.assertEqual(ledger[0]['repair_attempts'], 3)
        self.assertTrue(ledger[0]['unresolved'])
        final_hash = fingerprint([asdict(r) for r in final])
        self.assertEqual(ledger[0]['final_draft_sha256'], final_hash)
        self.assertEqual(ledger[0]['resolution_history'][-1]['assembled_draft_sha256'], final_hash)
        run = next(iter(self.saved()['runs'].values()))
        self.assertEqual(run['unresolved_issue_ids'], ['cue-1'])
        self.assertFalse(run['local_resolution_pass'])

    def test_unsupported_diagnosis_can_withdraw_unchanged_text(self):
        final, ledger, call = self.invoke([diagnosis('ISSUE', 'OK'), encoded({'1': '甲句。'}),
                                         resolution('diagnosis_unsupported')])
        self.assertEqual(final, self.draft)
        self.assertEqual(len(call.call_args_list), 3)
        self.assertEqual(ledger[0]['resolution_status'], 'diagnosis_unsupported')
        self.assertFalse(ledger[0]['unresolved'])
        self.assertEqual(ledger[0]['repair_attempts'], 1)

    def test_prior_resolved_and_withdrawn_issues_are_rechecked_on_each_assembled_draft(self):
        answers = [diagnosis('ISSUE', 'ISSUE'), encoded({'1': '甲句。', '2': '乙句。'}),
                   resolution('diagnosis_unsupported', 'persists'), encoded({'1': '更新乙句。'}),
                   resolution('persists', 'resolved'), encoded({'1': '更新甲句。'}),
                   resolution('resolved', 'persists')]
        final, ledger, call = self.invoke(answers)
        checks = [c for c in call.call_args_list if c.args[2].stage.endswith('_check')]
        self.assertEqual(len(checks), 3)
        for check in checks:
            self.assertEqual(set(json.loads(check.args[0])['items']), {'1', '2'})
        final_hash = fingerprint([asdict(r) for r in final])
        self.assertTrue(all(r['resolution_history'][-1]['assembled_draft_sha256'] == final_hash for r in ledger))
        self.assertTrue(ledger[1]['unresolved'])
        self.assertEqual([r['repair_attempts'] for r in ledger], [2, 2])
        final_check = json.loads(checks[-1].args[0])
        self.assertEqual(final_check['current_draft_sha256'], final_hash)
        self.assertIn('更新甲句。', checks[-1].args[1])
        self.assertIn('更新乙句。', checks[-1].args[1])

    def test_malformed_or_incomplete_resolution_does_not_return_candidate(self):
        for bad in (encoded({}), encoded({'2': {'status': 'resolved', 'reason': 'x'}}),
                    encoded({'1': {'status': 'approved', 'reason': 'x'}}),
                    '{"1":{"status":"resolved","reason":"x"},"1":{"status":"persists","reason":"x"}}'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temporary:
                self.cache = Path(temporary) / 'requests.json'
                with self.assertRaises(CoherenceError):
                    self.invoke([diagnosis('ISSUE', 'OK'), encoded({'1': '甲句。'}), bad, bad])
                self.assertFalse(next(iter(self.saved()['runs'].values()))['complete'])
        self.assertEqual(self.draft[0].text, '甲句。')

    def test_missing_repair_id_cannot_drop_cue(self):
        with self.assertRaises(CoherenceError):
            self.invoke([diagnosis('ISSUE', 'ISSUE'), encoded({'1': '甲句。'}), encoded({'1': '甲句。'})])
        self.assertFalse(next(iter(self.saved()['runs'].values()))['complete'])

    def test_cache_reparsed_and_checksum_mutation_rejected(self):
        answers = [diagnosis('ISSUE', 'OK'), encoded({'1': '甲句。'}), resolution('resolved')]
        expected, _, _ = self.invoke(answers)
        cached, _, call = self.invoke([])
        self.assertEqual(cached, expected)
        call.assert_not_called()
        saved = self.saved()
        run = next(iter(saved['runs'].values()))
        checked = next(entry for entry in run['batches'].values() if entry['request']['stage'].endswith('_check'))
        checked['response'] = encoded({})
        checked['response_sha256'] = _text_hash(checked['response'])
        self.cache.write_text(encoded(saved), encoding='utf-8')
        _, ledger, call = self.invoke([resolution('diagnosis_unsupported')])
        self.assertEqual(len(call.call_args_list), 1)
        self.assertEqual(ledger[0]['resolution_status'], 'diagnosis_unsupported')
        saved = self.saved()
        checked = next(entry for entry in next(iter(saved['runs'].values()))['batches'].values()
                       if entry['request']['stage'].endswith('_check'))
        self.assertTrue(checked['cache_rejections'])
        checked['response_sha256'] = 'incorrect'
        self.cache.write_text(encoded(saved), encoding='utf-8')
        _, _, call = self.invoke([resolution('resolved')])
        self.assertEqual(len(call.call_args_list), 1)

    def test_new_initial_draft_cannot_reuse_prior_requests(self):
        self.invoke([diagnosis('OK', 'OK')])
        self.draft[0] = replace(self.draft[0], text='另一甲句。')
        _, _, call = self.invoke([diagnosis('OK', 'OK')])
        self.assertEqual(len(call.call_args_list), 1)
        self.assertEqual(len(self.saved()['runs']), 2)

    def test_alignment_failure_before_any_request(self):
        self.draft[0] = replace(self.draft[0], ts_line='00:00:00,100 --> 00:00:02,000')
        with self.assertRaisesRegex(ValueError, 'must match exactly'):
            self.invoke([])
        self.capacity.assert_not_called()

    def test_source_and_external_notes_never_enter_requests(self):
        self.config.line_notes = {1: ['EXTERNAL_SECRET']}
        self.config.entity_notes = {1: ['EXTERNAL_SECRET']}
        before = deepcopy((self.source, self.draft))
        _, _, call = self.invoke([diagnosis('ISSUE', 'OK'), encoded({'1': '甲句。'}), resolution('resolved')])
        for request in call.call_args_list:
            body, instruction, cfg = request.args
            self.assertNotIn('SOURCE_SECRET', body + instruction)
            self.assertNotIn('EXTERNAL_SECRET', body + instruction)
            self.assertIn('global_id', instruction)
            self.assertIn(self.draft[1].ts_line, instruction)
            self.assertTrue(cfg.separate_instruction)
            self.assertEqual(cfg.retries, 0)
        self.assertEqual((self.source, self.draft), before)

    def test_unknown_related_context_id_rejected(self):
        bad = encoded({'1': {'status': 'ISSUE', 'reason': 'x', 'context_ids': [99]},
                       '2': {'status': 'OK', 'reason': '', 'context_ids': []}})
        with self.assertRaises(CoherenceError):
            self.invoke([bad, bad])

    def test_capacity_failure_precedes_generation_without_truncation(self):
        self.capacity.return_value = {'fits': False, 'prompt_tokens': 99999}
        with patch('src.coherence_refine.call_llm') as call, self.assertRaisesRegex(CoherenceError, 'context capacity'):
            R.refine_resolution(self.source, self.draft, self.config, self.cache, critic_budget=128, edit_budget=128)
        call.assert_not_called()


class CapacityChecks(unittest.TestCase):
    def test_exact_native_template_and_special_token_counting(self):
        cfg = TranslateConfig(endpoint='http://127.0.0.1:18101/v1/chat/completions', max_tokens=512,
            separate_instruction=True, extra_payload={'model': 'synthetic-local-writer', 'max_tokens': 512})
        with patch.object(R, '_request_json', side_effect=[{'default_generation_settings': {'n_ctx': 4096}},
              {'prompt': 'synthetic rendered prompt'}, {'tokens': [1, 2, 3]}]) as request:
            receipt = R.capacity_check('body', 'instruction', cfg, context_size=4096, thinking=False)
        self.assertTrue(receipt['fits'])
        self.assertEqual(receipt['reserved_completion_tokens'], 512)
        self.assertEqual(receipt['prompt_tokens'], 3)
        expected_payload = R._build_payload('body', 'instruction', cfg, R._THINKING_OFF_VARIANTS[0],
                                            512, stream=True, with_thinking=False)
        self.assertEqual(request.call_args_list[1].args[1], expected_payload)
        self.assertEqual(receipt['request_sha256'], fingerprint(expected_payload))
        self.assertEqual(request.call_args_list[-1].args[1],
                         {'content': 'synthetic rendered prompt', 'add_special': True, 'parse_special': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
