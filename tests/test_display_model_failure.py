"""Synthetic coverage for opt-in layout failure without altering cue content."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import requests

from src.config import TranslateConfig
from src.display import build_display
from src.translate import LLMRetryExhaustedError, RunawayError, SrtBlock, call_llm


class DisplayModelFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / 'layout.json'
        self.source = [SrtBlock(1, '00:00:01,123 --> 00:00:11,456', 'source-alpha.source-beta.')]
        self.target = [SrtBlock(1, self.source[0].ts_line, 'target-placeholder.' * 3)]
        self.metadata = {'utterance_tokens': [[dict(text=self.source[0].text, start=2., end=9.)]]}
        self.cfg = TranslateConfig(retries=0)

    def build(self, enabled=True):
        return build_display(self.source, self.target, self.metadata, self.cfg, self.cache,
                             model_failure_fallback=enabled)

    def test_retry_exhaustion_has_specific_type_and_cause(self):
        cause = RunawayError('synthetic output guard')
        with patch('src.translate._stream_response', side_effect=cause), patch('src.translate._record_request'):
            with self.assertRaises(LLMRetryExhaustedError) as caught:
                call_llm('synthetic', 'synthetic', self.cfg)
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIsInstance(caught.exception, RuntimeError)

    def test_rejected_request_is_not_retry_exhaustion(self):
        response = requests.Response()
        response.status_code = 422
        failure = requests.HTTPError('synthetic schema rejection', response=response)
        with patch('src.translate._stream_response', side_effect=failure), patch('src.translate._record_request'):
            with self.assertRaises(RuntimeError) as caught:
                call_llm('synthetic', 'synthetic', self.cfg)
        self.assertNotIsInstance(caught.exception, LLMRetryExhaustedError)

    def test_failure_stays_fatal_by_default(self):
        with patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
            with self.assertRaises(LLMRetryExhaustedError):
                build_display(self.source, self.target, self.metadata, self.cfg, self.cache)

    def test_fallback_preserves_exact_text_timestamp_and_input(self):
        original = deepcopy((self.source, self.target, self.metadata))
        with patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
            with self.assertLogs('src.display', level='WARNING'):
                raw, final, rows = self.build()
        self.assertEqual((self.source, self.target, self.metadata), original)
        self.assertEqual(raw, self.source)
        self.assertEqual(final, self.target)
        self.assertEqual(rows[0]['start'], 1.123)
        self.assertEqual(rows[0]['end'], 11.456)
        self.assertTrue(rows[0]['layout_warning']['review_required'])
        self.assertEqual(rows[0]['layout_warning']['stage'], 'display-layout')
        self.assertFalse(self.cache.exists())
        with patch('src.display.call_llm', return_value=json.dumps({'parts': [dict(source=self.source[0].text, target=self.target[0].text)]})) as call:
            self.build()
        call.assert_called_once()  # A failed request is not a successful cache entry.

    def test_pause_failure_preserves_whole_cue_and_does_not_try_second_planner(self):
        self.source[0].text = 'alpha.bravo.'
        self.target[0].text = 'first.second.'
        self.metadata = {'utterance_tokens': [[dict(text='alpha.', start=2., end=3.),
                                              dict(text='bravo.', start=6., end=8.)]],
                         'speech_regions': [dict(start=2000, end=3000), dict(start=6000, end=8000)],
                         'sample_rate': 1000}
        self.cfg = TranslateConfig(pause_layout=True)
        with patch('src.pause_layout.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
            with patch('src.display.call_llm') as other:
                raw, final, rows = self.build()
        other.assert_not_called()
        self.assertEqual((raw, final), (self.source, self.target))
        self.assertEqual(rows[0]['pause_layout']['status'], 'fallback_model_failure')
        self.assertEqual(rows[0]['layout_warning']['stage'], 'pause-display-layout')
        self.assertFalse(self.cache.with_suffix('.pauses.json').exists())

    def test_neighbor_holds_respect_preserved_interval(self):
        times = ['00:00:00,000 --> 00:00:01,000', '00:00:02,123 --> 00:00:12,456',
                 '00:00:13,000 --> 00:00:14,000']
        self.source = [SrtBlock(i, ts, 'source') for i, ts in enumerate(times, 1)]
        self.target = [SrtBlock(i, ts, 'target' if i != 2 else 'target' * 8) for i, ts in enumerate(times, 1)]
        self.metadata = {}
        with patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
            raw, final, rows = self.build()
        self.assertEqual(final[1], self.target[1])
        self.assertEqual(raw[1], self.source[1])
        self.assertLessEqual(rows[0]['end'], rows[1]['start'])
        self.assertLessEqual(rows[1]['end'], rows[2]['start'])
        self.assertEqual([r['utterance'] for r in rows], [1, 2, 3])

    def test_overlap_is_not_silently_trimmed_to_enable_fallback(self):
        for times in (['00:00:00,000 --> 00:00:03,000', '00:00:02,000 --> 00:00:12,000'],
                      ['00:00:01,000 --> 00:00:11,000', '00:00:10,000 --> 00:00:12,000']):
            with self.subTest(times=times):
                self.source = [SrtBlock(i, ts, 'source') for i, ts in enumerate(times, 1)]
                self.target = [SrtBlock(i, ts, 'target' * 8) for i, ts in enumerate(times, 1)]
                self.metadata = {}
                with patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
                    with self.assertRaisesRegex(ValueError, 'overlapping'):
                        self.build()

    def test_other_errors_and_cache_write_failures_remain_fatal(self):
        for failure in (RuntimeError('hash guard'), ValueError('schema guard'), OSError('storage'), KeyboardInterrupt()):
            with self.subTest(kind=type(failure).__name__):
                with patch('src.display.call_llm', side_effect=failure):
                    with self.assertRaises(type(failure)):
                        self.build()
        valid = json.dumps({'parts': [dict(source=self.source[0].text, target=self.target[0].text)]})
        with patch('src.display.call_llm', return_value=valid), patch('src.display.write_json', side_effect=RuntimeError('cache write guard')):
            with self.assertRaisesRegex(RuntimeError, 'cache write guard'):
                self.build()

    def test_mismatched_pair_or_invalid_geometry_still_blocks(self):
        original = deepcopy((self.source, self.target, self.metadata))
        for kind in ('ids', 'timestamps', 'empty', 'reversed', 'token_missing', 'token_nonfinite'):
            self.source, self.target, self.metadata = deepcopy(original)
            if kind == 'ids': self.target[0].index = 2
            if kind == 'timestamps': self.target[0].ts_line = '00:00:02,000 --> 00:00:12,000'
            if kind == 'empty': self.source[0].text = ''
            if kind == 'reversed':
                self.source[0].ts_line = self.target[0].ts_line = '00:00:11,456 --> 00:00:01,123'
            if kind == 'token_missing': del self.metadata['utterance_tokens'][0][0]['start']
            if kind == 'token_nonfinite': self.metadata['utterance_tokens'][0][0]['end'] = float('nan')
            with self.subTest(kind=kind), patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
                with self.assertRaises((ValueError, KeyError)):
                    self.build()

    def test_later_metadata_is_not_skipped_after_a_fallback(self):
        self.source.append(SrtBlock(2, '00:00:13,000 --> 00:00:14,000', 'other'))
        self.target.append(SrtBlock(2, self.source[1].ts_line, 'other'))
        self.metadata['utterance_tokens'].append([{}])
        with patch('src.display.call_llm', side_effect=LLMRetryExhaustedError('synthetic')):
            with self.assertRaises(KeyError):
                self.build()


if __name__ == '__main__':
    unittest.main()
