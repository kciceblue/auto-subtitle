"""Transactional source repair tests; no endpoint, audio or GPU is used."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import TranslateConfig
from src.source_realign import SourceRealignmentResult
from src.source_repair import repair_source_spans
from src.translate import SrtBlock
from src.workflow_state import fingerprint, read_json, write_json

Q = '明日の午後はエイガを見に行きます。'
WN = '明日の午後は映画を見に行きます。'


def fixture(count=1):
    metadata = {key: [] for key in ('windows', 'raw_transcripts', 'used_transcripts',
                                   'source_selection', 'segments', 'utterance_tokens')}
    source, decisions = [], []
    for i in range(count):
        uid, start, end = 17+i, i*10+1., i*10+8.
        raw = dict(q=Q, w=WN, n=WN)
        metadata['windows'].append(dict(index=i, start=i*10., end=i*10+10.,
                                       core_start=i*10., core_end=i*10+10.))
        metadata['raw_transcripts'].append(raw)
        metadata['used_transcripts'].append(deepcopy(raw))
        metadata['source_selection'].append(dict(window=i, selected='q'))
        metadata['segments'].append(dict(index=uid, text=Q, start=start, end=end))
        metadata['utterance_tokens'].append([dict(text=Q, begin=0, finish=len(Q), start=start, end=end)])
        timestamp = f'00:00:{int(start):02d},000 --> 00:00:{int(end):02d},000'
        source.append(SrtBlock(uid, timestamp, Q))
        decisions.append(dict(line=uid, text=Q, status='unresolved', candidate='Q',
                              reason='Original acoustic uncertainty', evidence={'raw': Q, 'grade': 'C'}))
    return source, decisions, metadata


def choose(body, instruction, config):
    tasks = json.loads(body)
    return json.dumps({key: dict(decision=task['task']['candidate_spans'][0]['id'],
                                reason='Both positional raw alternatives agree')
                       for key, task in tasks.items()})


def keep(body, instruction, config):
    return json.dumps({key: dict(decision='KEEP', reason='Keep raw evidence unresolved')
                       for key in json.loads(body)})


def aligned(media, source, changed_ids, metadata, config, output_path, failed_ids=()):
    copied = deepcopy(metadata)
    successes, failures = [], {}
    for i, row in enumerate(source):
        if row.index not in changed_ids:
            continue
        if row.index in failed_ids:
            failures[row.index] = 'Missing acoustic coverage'
            continue
        copied['utterance_tokens'][i] = [dict(text=row.text, begin=0, finish=len(row.text),
            start=metadata['segments'][i]['start'], end=metadata['segments'][i]['end'],
            source_id=row.index, source_hash=fingerprint(row.text), offset_scope='utterance')]
        successes.append(row.index)
    report = dict(request_key='matched-alignment', successful_ids=successes, failed_ids=failures)
    return SourceRealignmentResult(copied, tuple(successes), failures, report, output_path)


class SourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root/'original.mp4'
        self.media.write_bytes(b'local audio fixture; never decoded')
        self.cache = self.root/'source-selection.json'
        self.alignment = self.root/'source-realignment.json'
        self.config = TranslateConfig(endpoint='http://127.0.0.1:8089/v1/chat/completions',
            extra_payload={'model': 'qwen3.8-27b-dflash', 'temperature': .3},
            context_summary='Provided background only')

    def run_repair(self, data=None, **kwargs):
        source, decisions, metadata = data or fixture()
        return repair_source_spans(self.media, source, decisions, metadata,
            kwargs.pop('config', self.config), self.cache, self.alignment, **kwargs)

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_success_preserves_all_original_evidence_and_changes_only_aligned_text(self, llm, align):
        data = fixture()
        original = deepcopy(data)
        result = self.run_repair(data)
        self.assertTrue(result.complete)
        self.assertEqual(result.source[0].text, WN)
        self.assertEqual(result.source[0].ts_line, data[0][0].ts_line)
        self.assertEqual(data, original)
        self.assertEqual(result.decisions[0]['status'], 'unresolved')
        self.assertEqual(result.decisions[0]['evidence'], original[1][0]['evidence'])
        self.assertEqual(result.decisions[0]['source_span_repair']['prior_decision'], original[1][0])
        self.assertEqual(result.metadata['segments'], original[2]['segments'])
        self.assertEqual(result.metadata['raw_transcripts'], original[2]['raw_transcripts'])
        self.assertEqual(result.report['changed_ids'], [17])
        self.assertEqual(result.report['patches'][0]['status'], 'realigned_unresolved')
        self.assertEqual(align.call_args.args[2], [17])
        self.assertEqual(llm.call_args.args[2].retries, 0)
        self.assertEqual(llm.call_args.args[2].max_tokens, 2048)
        self.assertFalse(llm.call_args.args[2].extra_payload['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(self.media.read_bytes(), b'local audio fixture; never decoded')

    @patch('src.source_repair.realign_changed')
    @patch('src.source_repair.call_llm', side_effect=keep)
    def test_keep_retains_doubts_without_invoking_aligner(self, llm, align):
        result = self.run_repair()
        self.assertTrue(result.complete)
        self.assertEqual(result.source[0].text, Q)
        self.assertEqual(result.decisions[0]['status'], 'unresolved')
        self.assertEqual(result.decisions[0]['reason'], 'Original acoustic uncertainty')
        align.assert_not_called()

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_exact_cached_response_reused_but_realignment_still_required(self, llm, align):
        first = self.run_repair()
        second = self.run_repair()
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(align.call_count, 2)
        self.assertTrue(second.report['batches'][0]['cache_hit'])
        self.assertEqual(first.source, second.source)
        data = read_json(self.cache)
        entry = next(iter(data['batches'].values()))
        self.assertEqual(entry['answer'], entry['attempts'][-1]['response'])

    @patch('src.source_repair.realign_changed')
    @patch('src.source_repair.call_llm', side_effect=keep)
    def test_cache_binds_model_context_policy_locks_and_original_metadata(self, llm, align):
        self.run_repair()
        self.run_repair()
        self.assertEqual(llm.call_count, 1)
        self.run_repair(config=replace(self.config, context_summary='Changed supplied background'))
        self.run_repair(config=replace(self.config, extra_payload={'model': 'qwen3.8-27b-orca'}))
        self.run_repair(protected_surfaces=['別名'])
        data = fixture()
        data[2]['utterance_tokens'][0][0]['start'] += .01
        self.run_repair(data)
        with patch('src.source_repair.VERSION', 'source-repair-next'):
            self.run_repair(data)
        self.assertEqual(llm.call_count, 6)
        align.assert_not_called()

    @patch('src.source_repair.realign_changed')
    @patch('src.source_repair.call_llm', side_effect=keep)
    def test_tampered_cached_raw_answer_is_not_trusted(self, llm, align):
        self.run_repair()
        data = read_json(self.cache)
        next(iter(data['batches'].values()))['answer'] = '{"1":{"decision":"A001","reason":"tampered"}}'
        write_json(self.cache, data)
        self.run_repair()
        self.assertEqual(llm.call_count, 2)
        align.assert_not_called()

    @patch('src.source_repair.realign_changed')
    @patch('src.source_repair.call_llm', return_value='{}')
    def test_missing_whole_response_retries_once_and_remains_unchecked(self, llm, align):
        result = self.run_repair()
        self.assertEqual(llm.call_count, 2)
        self.assertFalse(result.complete)
        self.assertEqual(result.decisions[0]['status'], 'unchecked')
        self.assertEqual(result.source[0].text, Q)
        self.assertEqual(result.report['unchecked_ids'], [17])
        self.assertEqual([a['response'] for a in result.report['batches'][0]['attempts']], ['{}', '{}'])
        self.run_repair()
        self.assertEqual(llm.call_count, 4)  # Failed replies never become a KEEP cache.
        align.assert_not_called()

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    def test_valid_retry_records_failed_response_and_applies_fixed_alternative(self, align):
        calls = []
        def response(body, instruction, config):
            calls.append(instruction)
            return 'not JSON' if len(calls) == 1 else choose(body, instruction, config)
        with patch('src.source_repair.call_llm', side_effect=response):
            result = self.run_repair()
        self.assertTrue(result.complete)
        self.assertEqual(len(calls), 2)
        self.assertIn('previous response failed', calls[1])
        self.assertEqual(result.report['batches'][0]['attempts'][0]['response'], 'not JSON')
        self.assertEqual(result.source[0].text, WN)

    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_partial_alignment_failure_rolls_back_failed_source_and_tokens(self, llm):
        data = fixture(2)
        def partial(*args):
            return aligned(*args, failed_ids=(18,))
        with patch('src.source_repair.realign_changed', side_effect=partial):
            result = self.run_repair(data)
        self.assertFalse(result.complete)
        self.assertEqual([row.text for row in result.source], [WN, Q])
        self.assertEqual(result.metadata['utterance_tokens'][1], data[2]['utterance_tokens'][1])
        self.assertEqual(result.report['failed_alignment_ids'], {'18': 'Missing acoustic coverage'})
        self.assertEqual([p['status'] for p in result.report['patches']],
                         ['realigned_unresolved', 'reverted_unresolved'])
        self.assertEqual([row['status'] for row in result.decisions], ['unresolved', 'unresolved'])

    @patch('src.source_repair.call_llm', side_effect=choose)
    @patch('src.source_repair.realign_changed', side_effect=RuntimeError('worker unavailable'))
    def test_worker_failure_reverts_every_projected_row(self, align, llm):
        data = fixture(2)
        result = self.run_repair(data)
        self.assertFalse(result.complete)
        self.assertEqual(result.source, data[0])
        self.assertEqual(result.metadata, data[2])
        self.assertEqual(result.report['changed_ids'], [])
        self.assertEqual(set(result.report['failed_alignment_ids']), {'17', '18'})

    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_stale_success_tokens_do_not_commit_source(self, llm):
        def stale(*args):
            result = aligned(*args)
            result.metadata['utterance_tokens'][0][0]['source_hash'] = 'different-source'
            result.metadata['raw_transcripts'] = ['do not import mutated raw evidence']
            return result
        with patch('src.source_repair.realign_changed', side_effect=stale):
            result = self.run_repair()
        self.assertFalse(result.complete)
        self.assertEqual(result.source[0].text, Q)
        self.assertEqual(result.metadata, fixture()[2])

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    def test_batches_are_three_windows_and_failed_batch_does_not_block_valid_batch(self, align):
        sizes = []
        def response(body, instruction, config):
            sizes.append(len(json.loads(body)))
            return '{}' if len(json.loads(body)) == 3 else choose(body, instruction, config)
        with patch('src.source_repair.call_llm', side_effect=response):
            result = self.run_repair(fixture(4))
        self.assertEqual(sizes, [3, 3, 1])
        self.assertFalse(result.complete)
        self.assertEqual([row.text for row in result.source], [Q, Q, Q, WN])
        self.assertEqual(align.call_args.args[2], [20])
        self.assertEqual(result.report['unchecked_ids'], [17, 18, 19])

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_prior_unchecked_source_confidence_is_not_upgraded(self, llm, align):
        data = fixture()
        data[1][0]['status'] = 'unchecked'
        result = self.run_repair(data)
        self.assertEqual(result.decisions[0]['status'], 'unchecked')
        self.assertEqual(result.source[0].text, WN)

    @patch('src.source_repair.realign_changed')
    @patch('src.source_repair.call_llm')
    def test_locked_source_has_no_candidates_and_needs_no_inference(self, llm, align):
        result = self.run_repair(protected_surfaces=['エイガ'])
        self.assertTrue(result.complete)
        self.assertEqual(result.report['candidate_count'], 0)
        self.assertEqual(result.source, fixture()[0])
        llm.assert_not_called()
        align.assert_not_called()

    @patch('src.source_repair.realign_changed', side_effect=aligned)
    @patch('src.source_repair.call_llm', side_effect=choose)
    def test_hard_conflict_blocks_only_affected_source_before_alignment(self, llm, align):
        data = fixture(2)
        data[1][0]['evidence']['hard_conflict'] = True
        original = deepcopy(data)
        result = self.run_repair(data)
        self.assertTrue(result.complete)
        self.assertEqual(align.call_args.args[2], [18])
        self.assertEqual([row.text for row in result.source], [Q, WN])
        self.assertEqual(result.metadata['utterance_tokens'][0], data[2]['utterance_tokens'][0])
        self.assertEqual(result.decisions[0]['evidence'], data[1][0]['evidence'])
        self.assertEqual(result.decisions[0]['status'], 'unresolved')
        self.assertEqual(result.report['hard_conflict_blocked_ids'], [17])
        self.assertFalse(result.report['patches'][0]['applied'])
        self.assertIn('hard script/audio conflict', result.report['patches'][0]['rejection'])
        self.assertEqual(data, original)

    @patch('src.source_repair.realign_changed')
    def test_media_changed_during_selection_keeps_all_original_source_and_tokens(self, align):
        def replace_media(body, instruction, config):
            self.media.write_bytes(b'changed original media')
            return choose(body, instruction, config)
        data = fixture()
        with patch('src.source_repair.call_llm', side_effect=replace_media):
            result = self.run_repair(data)
        self.assertFalse(result.complete)
        self.assertEqual(result.source, data[0])
        self.assertEqual(result.metadata, data[2])
        self.assertEqual(result.decisions[0]['status'], 'unchecked')
        self.assertIn('media changed', result.report['input_error'])
        align.assert_not_called()

    @patch('src.source_repair.call_llm')
    def test_external_or_stronger_writer_is_rejected_before_inference(self, llm):
        for config in (replace(self.config, endpoint='https://external.invalid/v1/chat/completions'),
                       replace(self.config, extra_payload={'model': 'gpt-6-astra'}),
                       replace(self.config, extra_payload={'messages': []})):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_repair(config=config)
        llm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
