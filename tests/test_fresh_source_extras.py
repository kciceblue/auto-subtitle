"""Offline checks for fresh source-only acquisition; no model or GPU calls."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from src import fresh_source_extras as worker


def automatic_rows():
    return [
        {'owner_id': 1, 'text': 'recognized', 'status': 'ok', 'detected_language': 'English', 'vad_overlap_seconds': 1.0},
        {'owner_id': 2, 'text': 'recognized', 'status': 'ok', 'detected_language': 'Japanese', 'vad_overlap_seconds': 1.0},
        {'owner_id': 3, 'text': '', 'status': 'empty', 'detected_language': 'English', 'vad_overlap_seconds': 1.0},
        {'owner_id': 4, 'text': 'recognized', 'status': 'ok', 'detected_language': 'English', 'vad_overlap_seconds': 0.0},
        {'owner_id': 5, 'text': 'recognized', 'status': 'ok', 'detected_language': None, 'vad_overlap_seconds': 1.0},
    ]


class GeometryTests(unittest.TestCase):
    def test_arbitrary_owner_count_retains_complete_physical_domain(self):
        owners = [{'id': i+1, 'start': i*2, 'end': (i+1)*2} for i in range(67)]
        geometry = worker.owner_geometry(owners, 134*worker.RATE)
        self.assertEqual(len(geometry), 67)
        self.assertEqual(geometry[-1]['end_frame'], 134*worker.RATE)

    def test_gaps_overlaps_nonfinite_and_duplicate_owners_rejected(self):
        baseline = [{'id': 1, 'start': 0, 'end': 2}, {'id': 2, 'start': 2, 'end': 4}]
        for key, value in [('start', 2.1), ('start', 1.9), ('end', float('nan')), ('id', 1), ('id', True)]:
            rows = deepcopy(baseline)
            rows[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                worker.owner_geometry(rows, 4*worker.RATE)

    def test_incomplete_tail_and_overlong_owner_rejected(self):
        for owner, frames in [({'id': 1, 'start': 0, 'end': 1}, 2*worker.RATE),
                              ({'id': 1, 'start': 0, 'end': 31}, 31*worker.RATE)]:
            with self.assertRaises(ValueError):
                worker.owner_geometry([owner], frames)

    def test_submillisecond_tail_clamps_only_last_owner(self):
        result = worker.owner_geometry([{'id': 1, 'start': 0, 'end': 2}], 2*worker.RATE-1)
        self.assertEqual(result[0]['end_frame'], 2*worker.RATE-1)
        result = worker.owner_geometry([{'id': 1, 'start': 0, 'end': 2}], 2*worker.RATE+2)
        self.assertEqual(result[0]['end_frame'], 2*worker.RATE+2)
        with self.assertRaises(ValueError):
            worker.owner_geometry([{'id': 1, 'start': 0, 'end': 2}], 2*worker.RATE-9)

    def test_vad_intersections_preserve_empty_speech(self):
        geometry = worker.owner_geometry([{'id': 1, 'start': 0, 'end': 2},
            {'id': 2, 'start': 2, 'end': 4}], 4*worker.RATE)
        self.assertEqual(worker.vad_overlaps(geometry, [], 4*worker.RATE), {1: 0, 2: 0})
        self.assertEqual(worker.vad_overlaps(geometry, [{'start': worker.RATE, 'end': 3*worker.RATE}],
            4*worker.RATE), {1: 1, 2: 1})
        with self.assertRaises(ValueError):
            worker.vad_overlaps(geometry, [{'start': 10, 'end': 20}, {'start': 19, 'end': 30}], 4*worker.RATE)


class EligibilityTests(unittest.TestCase):
    def test_only_nonempty_non_japanese_with_speech_is_eligible(self):
        self.assertEqual(worker.eligible_owner_ids(automatic_rows()), [1])
        self.assertEqual(worker.eligible_owner_ids([]), [])

    def test_unavailable_language_and_failed_calls_are_not_selected(self):
        rows = automatic_rows()
        rows[0]['status'] = 'unavailable'
        self.assertEqual(worker.eligible_owner_ids(rows), [])

    def test_duplicate_and_invalid_overlap_rejected(self):
        rows = automatic_rows()
        with self.assertRaises(ValueError):
            worker.eligible_owner_ids(rows+[rows[0]])
        for overlap in (None, -1, float('inf'), True):
            rows[0]['vad_overlap_seconds'] = overlap
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                worker.eligible_owner_ids(rows)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mono = self.root/'mono.wav'
        sf.write(self.mono, np.zeros(4*worker.RATE, dtype=np.float32), worker.RATE, subtype='FLOAT')
        self.spec = {'method': 'auto', 'mono_path': str(self.mono),
            'owners': [{'id': 1, 'start': 0, 'end': 2}, {'id': 2, 'start': 2, 'end': 4}],
            'output_dir': str(self.root/'output'), 'model_dir': str(self.root/'model'),
            'vad_speech': [{'start': 0, 'end': worker.RATE}], 'worker_seconds': 30}
        self.path = self.root/'spec.json'
        model_file = self.root/'model.bin'
        model_file.write_bytes(b'synthetic pinned model')
        self.assets = {'model': {'files': [worker.pin(model_file)]}}

    def write_spec(self):
        self.path.write_text(json.dumps(self.spec))

    def fake_capture(self, method, session, request, assets, folder, deadline, save_capture, save_attention):
        save_capture({'raw_text': '', 'test_only': True})
        save_attention(None)
        text = 'source' if request['kind'] == 'real' and request['owner_id'] == 1 else ''
        return {'text': text, 'detected_language': 'Japanese', 'raw_text': text}

    def run_fake(self, capture=None):
        self.write_spec()
        with patch.object(worker, 'load_session', return_value=(self.assets, object())) as load, \
             patch.object(worker, 'build_request', side_effect=lambda method, slot, assets, prep: {
                 'kind': slot['kind'], 'owner_id': slot['owner_id'], 'audio': slot['audio']}), \
             patch.object(worker, 'capture_one', side_effect=capture or self.fake_capture):
            result = worker.run(self.path)
        return result, load.call_count

    def test_unknown_fields_reject_reference_or_context_contamination(self):
        for key in ('context', 'chinese', 'reference_srt', 'source_text'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_spec({**self.spec, key: 'unexpected'})

    def test_preparation_preserves_all_owners_and_deterministic_controls(self):
        folder = self.root/'prepared'
        folder.mkdir()
        result = worker.prepare_inputs(self.spec, folder)
        self.assertEqual([s['slot_id'] for s in result['slots']],
            ['silence', 'noise', 'owner-0001', 'owner-0002'])
        self.assertEqual(result['slots'][-1]['vad_overlap_seconds'], 0)
        self.assertEqual(result['gain'], 1)

    def test_one_session_captures_empty_owners_and_binds_receipts(self):
        result, loads = self.run_fake()
        self.assertTrue(result['complete'])
        self.assertEqual(loads, 1)
        self.assertEqual(result['asr_calls'], 4)
        self.assertEqual([r['status'] for r in result['observations']], ['ok', 'empty'])
        for row in result['observations']:
            self.assertEqual(row['receipt_sha256'], worker.pin(row['receipt_path'])['sha256'])
        self.assertFalse(result['backend_managed_by_worker'])

    def test_repeat_attempt_cannot_load_or_reroll(self):
        self.run_fake()
        with patch.object(worker, 'load_session') as load, self.assertRaises(ValueError):
            worker.run(self.path)
        load.assert_not_called()

    def test_control_hallucination_stops_before_real_source(self):
        def bad_control(*args):
            return {'text': 'hallucination', 'detected_language': 'Japanese', 'raw_text': 'hallucination'}
        result, loads = self.run_fake(bad_control)
        self.assertFalse(result['complete'])
        self.assertEqual(loads, 1)
        self.assertEqual(result['asr_calls'], 1)
        self.assertEqual(result['observations'], [])

    def test_failure_preserves_attempt_and_never_retries(self):
        calls = []
        def failure(*args):
            calls.append(True)
            args[-2]({'partial_native_output': [1, 2, 3]})
            raise RuntimeError('synthetic failure')
        result, _ = self.run_fake(failure)
        self.assertFalse(result['complete'])
        self.assertEqual(len(calls), 1)
        receipt = worker.read(self.root/'output/receipts/silence.json')
        self.assertEqual(receipt['capture']['partial_native_output'], [1, 2, 3])

    def test_forced_empty_selection_skips_loading(self):
        automatic, _ = self.run_fake()
        self.spec.update(method='forced', output_dir=str(self.root/'forced'),
            automatic_evidence=str(self.root/'output/evidence.json'), eligible_ids=[])
        self.write_spec()
        with patch.object(worker, 'load_session') as load:
            result = worker.run(self.path)
        load.assert_not_called()
        self.assertTrue(result['complete'])
        self.assertEqual(result['status'], 'skipped_no_eligible_owners')
        self.assertEqual(result['asr_calls'], 0)

    def test_forced_selection_rejects_cross_recording_or_cherry_picked_ids(self):
        self.run_fake()
        self.spec.update(method='forced', output_dir=str(self.root/'forced'),
            automatic_evidence=str(self.root/'output/evidence.json'), eligible_ids=[1])
        self.write_spec()
        with patch.object(worker, 'load_session') as load:
            result = worker.run(self.path)
        load.assert_not_called()
        self.assertFalse(result['complete'])
        self.assertIn('eligibility', result['error'])

    def test_historical_qwen_policy_remains_512(self):
        with patch.object(worker.automatic, 'execution_identity', return_value={'reserved_tokens': []}):
            request = worker.build_request('auto', {'owner_id': 67, 'audio': {}}, {}, {})
        self.assertEqual(request['policy']['max_new_tokens'], 512)
        self.assertIsNone(request['language'])
        self.assertEqual(request['conditioning'], {'context': '', 'language': None, 'hotwords': []})

    def test_real_truncation_stays_unavailable_and_remaining_owners_continue(self):
        calls = []
        def capture(*args):
            request = args[2]
            calls.append(request['owner_id'])
            if request['kind'] == 'real' and request['owner_id'] == 1:
                args[-2]({'output_token_ids': [1]*512, 'normal_stop': False})
                raise ValueError('Incomplete or truncated native call')
            return self.fake_capture(*args)
        result, loads = self.run_fake(capture)
        self.assertTrue(result['complete'])
        self.assertFalse(result['all_observations_available'])
        self.assertEqual(result['status'], 'complete_with_unavailable')
        self.assertEqual(calls, [None, None, 1, 2])
        self.assertEqual(result['observations'][0]['text'], '')
        self.assertIsNone(result['observations'][0]['detected_language'])
        self.assertEqual(result['counts'], {'ok': 0, 'empty': 1, 'unavailable': 1})
        self.assertEqual(loads, 1)

    def make_failed_original(self):
        def capture(*args):
            if args[2]['kind'] == 'real':
                args[-2]({'output_token_ids': [1]*512, 'normal_stop': False})
                raise ValueError('Incomplete or truncated native call')
            return self.fake_capture(*args)
        # Reproduce the v1 stop-on-first-error behavior, preserving its receipt.
        with patch.object(worker, 'recoverable_observation_error', return_value=False):
            result, _ = self.run_fake(capture)
        self.assertFalse(result['complete'])
        self.assertEqual(result['asr_calls'], 3)
        return result

    def continuation_spec(self):
        self.spec.update(output_dir=str(self.root/'continuation'),
            resume_from=worker.pin(self.root/'output/evidence.json'))
        # Never overwrite the original specification used in the first attempt.
        self.path = self.root/'continuation-spec.json'

    def test_continuation_inherits_failure_and_never_retries_attempted_owners(self):
        previous = self.make_failed_original()
        prior_receipt = Path(previous['observations'][0]['receipt_path']).read_bytes()
        self.continuation_spec()
        calls = []
        def capture(*args):
            calls.append(args[2]['owner_id'])
            return self.fake_capture(*args)
        result, loads = self.run_fake(capture)
        self.assertTrue(result['complete'], result['error'])
        self.assertEqual(calls, [2])
        self.assertEqual(loads, 1)
        self.assertEqual(result['observations'][0], previous['observations'][0])
        self.assertEqual(result['inherited_calls'], 3)
        self.assertEqual(result['additional_calls'], 1)
        self.assertEqual(result['asr_calls'], 4)
        self.assertEqual(result['inherited_seconds'], previous['total_seconds'])
        self.assertEqual(result['total_model_loads'], 2)
        self.assertEqual(Path(previous['observations'][0]['receipt_path']).read_bytes(), prior_receipt)

    def test_continuation_rejects_tampered_native_receipt_before_loading(self):
        previous = self.make_failed_original()
        path = Path(previous['observations'][0]['receipt_path'])
        receipt = worker.read(path)
        receipt['capture']['output_token_ids'][0] = 2
        path.write_text(json.dumps(receipt))
        self.continuation_spec()
        result, loads = self.run_fake()
        self.assertFalse(result['complete'])
        self.assertEqual(loads, 0)
        self.assertIn('hash changed', result['error'])

    def test_continuation_rejects_unaccounted_prior_attempt(self):
        self.make_failed_original()
        (self.root/'output/receipts/owner-0002.started.json').write_text('{}')
        self.continuation_spec()
        result, loads = self.run_fake()
        self.assertFalse(result['complete'])
        self.assertEqual(loads, 0)
        self.assertIn('Unaccounted', result['error'])

    def test_continuation_rejects_changed_model_assets(self):
        self.make_failed_original()
        (self.root/'model.bin').write_bytes(b'changed model')
        self.continuation_spec()
        result, loads = self.run_fake()
        self.assertFalse(result['complete'])
        self.assertEqual(loads, 0)
        self.assertIn('hash changed', result['error'])

    def test_chained_continuation_is_explicitly_rejected(self):
        previous = self.make_failed_original()
        previous['continuation'] = {'earlier_attempt': 'unsupported chain'}
        (self.root/'output/evidence.json').write_text(json.dumps(previous))
        self.continuation_spec()
        result, loads = self.run_fake()
        self.assertFalse(result['complete'])
        self.assertEqual(loads, 0)
        self.assertIn('Chained continuation is unsupported', result['error'])

    def test_systemic_and_deadline_errors_remain_terminal(self):
        for error in (RuntimeError('CUDA out of memory'), worker.qwen_base.WorkerDeadline('deadline'),
                      OSError('missing native artifact'), ValueError('Native normalization changed PCM'),
                      ValueError('Native context/prompt multiplicity changed'),
                      ValueError('Wrong ASR native backend/dtype/scope'), ValueError('unknown contract violation')):
            self.assertFalse(worker.recoverable_observation_error(error))
        self.assertTrue(worker.recoverable_observation_error(ValueError('Incomplete or truncated native call')))


if __name__ == '__main__':
    unittest.main()
