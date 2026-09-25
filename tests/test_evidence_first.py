"""Offline checks of the evidence-first workflow's text handling (no GPU, no model calls)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import evidence_first as ef


def reading(owner, text, start=0.0, end=20.0, status='ok', observer='zipformer-ja'):
    return {'owner_id': owner, 'observer': observer, 'start': start, 'end': end, 'text': text, 'status': status}


class PartChecks(unittest.TestCase):
    def test_default_bounds_cover_every_window_in_runs_of_twelve(self):
        self.assertEqual(ef.part_bounds(30), [(1, 12), (13, 24), (25, 30)])

    def test_explicit_bounds_must_tile_the_windows(self):
        self.assertEqual(ef.part_bounds(5, '1-3,4-5'), [(1, 3), (4, 5)])
        with self.assertRaises(ValueError):
            ef.part_bounds(5, '1-3,5-5')

    def test_part_text_lists_readings_then_slots(self):
        items = [{'key': '1.0', 'owner': 1, 'start': 0.72, 'end': 9.6, 'japanese': 'うん', 'draft': '嗯'}]
        readings = {1: [{'observer': 'qwen3-asr-1.7b', 'start': 0.0, 'end': 62.03, 'text': 'うん。'}]}
        self.assertEqual(ef.part_text(range(1, 3), readings, items), '\n'.join([
            '## WINDOW 1',
            '  reading qwen3-asr-1.7b [00:00.00-01:02.03]: うん。',
            '  SLOT 1.0 [00:00.72-00:09.60] JA: うん | DRAFT: 嗯',
            '',
            '## WINDOW 2',
            '  (no subtitle slots in this window)',
            '']))

    def test_readings_keep_first_ok_nonempty_text_per_window(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'base-observations.json').write_text(json.dumps(
                [reading(1, 'あ'), reading(1, 'あ', observer='qwen3-asr-1.7b'), reading(1, ''),
                 reading(2, 'い', status='empty')]), encoding='utf-8')
            files = {}
            for name in ef.EXTRAS:
                path = work / f'{name}.json'
                path.write_text(json.dumps({'observations': [reading(2, name)]}), encoding='utf-8')
                files[name] = {'path': str(path)}
            (work / 'extras-complete.json').write_text(json.dumps({'files': files}), encoding='utf-8')
            result = ef.readings_by_window(work)
        self.assertEqual([r['text'] for r in result[1]], ['あ'])
        self.assertEqual([r['text'] for r in result[2]], list(ef.EXTRAS))


class WriterChecks(unittest.TestCase):
    def test_writer_uses_reasoning_with_a_large_budget(self):
        config = ef.writer_config(Path('metrics.jsonl'))
        self.assertEqual(config.max_tokens, 32768)
        self.assertEqual(config.extra_payload['max_tokens'], 32768)
        self.assertTrue(config.extra_payload['chat_template_kwargs']['enable_thinking'])

    def test_answer_parser_accepts_common_key_forms_and_keeps_first(self):
        answer = '[1.0] 你好\nSLOT 1.1 走吧\n2.0\n[1.0] 重复\n说明文字'
        with mock.patch.object(ef, 'cached_call', return_value=answer):
            found = ef.write_part('body', ef.writer_config(Path('m.jsonl')), Path('cache.json'))
        self.assertEqual(found, {'1.0': '你好', '1.1': '走吧', '2.0': ''})

    def test_failed_reasoning_call_retries_once_without_thinking(self):
        calls = []

        def fake(body, config, cache):
            calls.append(config.extra_payload['chat_template_kwargs']['enable_thinking'])
            if len(calls) == 1:
                raise RuntimeError('empty content')
            return '[1.0] 好'
        with mock.patch.object(ef, 'cached_call', side_effect=fake):
            found = ef.write_part('body', ef.writer_config(Path('m.jsonl')), Path('cache.json'))
        self.assertEqual((calls, found), ([True, False], {'1.0': '好'}))


class BuildChecks(unittest.TestCase):
    def test_empty_slot_is_dropped_and_punctuation_convention_applied(self):
        units = [{'text': 'おはよう。', 'start': 1.0, 'end': 2.0}, {'text': 'はあ', 'start': 3.0, 'end': 3.5}]
        rows = [{'owner': 1, 'units': units, 'pieces': [{'units': [0, 0], 'target': '早。'},
                                                         {'units': [1, 1], 'target': '哈'}],
                 'source': {'ts_line': '00:00:00,000 --> 00:00:20,000'}, 'target': {'text': '早。哈'}}]
        cues = ef.build_cues(rows, {'1.0': '早上好，紬。', '1.1': ''})
        self.assertEqual([c['text'] for c in cues], ['早上好 紬'])
        self.assertGreaterEqual(cues[0]['start'], 1.0)

    def test_default_title_drops_release_tags(self):
        self.assertEqual(ef.default_title(Path('Show - S01E10 - Name HDTV-1080p.mp4')), 'Show - S01E10 - Name')
        self.assertEqual(ef.default_title(Path('[Group] Show 03 [1080p].mkv')), '[Group] Show 03')


if __name__ == '__main__':
    unittest.main()
