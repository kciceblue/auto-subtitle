"""Offline contracts for optional local editing, never a quality approval test."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.coherence import CoherenceError, MAX_CONTEXT_CHARS, MAX_DRAFT_CHARS, polish_coherence
from src.config import TranslateConfig
from src.translate import SrtBlock


class CoherenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root/'editor.json'
        self.source = [SrtBlock(i, f'00:00:0{2*i},000 --> 00:00:0{2*i+1},000', text)
                       for i, text in enumerate(('帰りましょう。', '待ってください。', 'はい。'), 1)]
        self.draft = [replace(row, text=text) for row, text in zip(self.source, ('回去吧。', '等一等。', '好。'))]
        self.context = self.root/'context.txt'
        self.context.write_text('原始背景资料。', encoding='utf-8')
        self.config = TranslateConfig(endpoint='http://127.0.0.1:8089/v1/chat/completions',
            context_files=[self.context], max_tokens=4096, retries=9,
            extra_payload={'model': 'local-editor-7b-q4', 'temperature': .3, 'seed': 42,
                           'max_tokens': 4096, 'chat_template_kwargs': {'enable_thinking': True}})

    def invoke(self, responses=None, **kwargs):
        answers = responses or ['{"1":"回去吧。","2":"请等一下。"}', '{"1":"好。"}']
        with patch('src.coherence.call_llm', side_effect=answers) as request:
            result = polish_coherence(self.source, self.draft, self.config, self.cache,
                                      batch_size=2, **kwargs)
        return result, request

    def saved(self):
        return json.loads(self.cache.read_text(encoding='utf-8'))

    def batch(self):
        run = next(iter(self.saved()['runs'].values()))
        return next(iter(run['batches'].values()))

    def test_lossless_srt_reader_preserves_display_boundary_spaces(self):
        from src.translate import parse_srt, write_translated_srt
        blocks = [replace(self.source[0], text=' leading \n trailing ')]
        path = self.root / 'display.srt'
        write_translated_srt(blocks, path)
        self.assertEqual(parse_srt(path, preserve_text_whitespace=True), blocks)
        self.assertEqual(parse_srt(path)[0].text, 'leading\ntrailing')

    def test_multiline_editor_result_survives_srt_write_and_parse(self):
        from src.translate import parse_srt, write_translated_srt
        responses = [json.dumps({'1': '回去吧。  \n  快走。', '2': '等一等。'}, ensure_ascii=False),
                     json.dumps({'1': '好。'}, ensure_ascii=False)]
        (final, _), _ = self.invoke(responses)
        path = self.root / 'candidate.zh.srt'
        write_translated_srt(final, path)
        self.assertEqual(parse_srt(path), final)
        self.assertEqual(final[0].text, '回去吧。\n快走。')

    def test_complete_original_context_and_immutable_draft_are_used_in_every_batch(self):
        background = '背景。'*2000+'最后一条原始资料'
        self.context.write_text(background, encoding='utf-8')
        self.config.context_summary = 'LOSSY SUMMARY MUST NOT REPLACE ORIGINAL'
        self.config.line_notes = {1: ['EVALUATOR FINDING MUST NEVER APPEAR']}
        self.config.entity_notes = {1: ['UNRELATED ENTITY NOTE MUST NEVER APPEAR']}
        before_source, before_draft, before_config = deepcopy((self.source, self.draft, self.config))
        (final, ledger), request = self.invoke()
        self.assertEqual((self.source, self.draft, self.config), (before_source, before_draft, before_config))
        self.assertEqual([row.index for row in final], [1, 2, 3])
        self.assertEqual([row.ts_line for row in final], [row.ts_line for row in self.source])
        self.assertEqual(final[1].text, '请等一下。')
        self.assertIsNot(final[0], self.draft[0])
        instructions = [call.args[1] for call in request.call_args_list]
        self.assertEqual(instructions[0], instructions[1])
        for instruction in instructions:
            self.assertIn(background, instruction)
            self.assertIn('回去吧。\n等一等。\n好。', instruction)
            self.assertNotIn('请等一下。', instruction)
            self.assertNotIn('MUST NEVER APPEAR', instruction)
            self.assertNotIn('LOSSY SUMMARY', instruction)
        last_body = json.loads(request.call_args_list[1].args[0])
        self.assertEqual(last_body, {'1': {'source': 'はい。', 'chinese': '好。'}})
        self.assertEqual([row['line'] for row in ledger], [1, 2, 3])
        self.assertTrue(all(row['status'] == 'UNVERIFIED' and not row['source_uncertainty_cleared'] for row in ledger))
        self.assertTrue(all(row['model'] == 'local-editor-7b-q4' and not row['cached'] for row in ledger))
        for call in request.call_args_list:
            cfg = call.args[2]
            self.assertEqual(cfg.retries, 0)
            self.assertEqual(cfg.extra_payload['seed'], 42)
            self.assertEqual(cfg.extra_payload['temperature'], .3)
            self.assertEqual(cfg.max_tokens, 4096)
            self.assertFalse(cfg.extra_payload['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(cfg.extra_payload['reasoning_effort'], 'none')
            schema = cfg.extra_payload['response_format']['schema']
            self.assertFalse(schema['additionalProperties'])
            self.assertEqual(set(schema['required']), set(json.loads(call.args[0])))

    def test_good_cache_reuses_reparsed_raw_responses(self):
        (first, _), _ = self.invoke()
        with patch('src.coherence.call_llm', side_effect=AssertionError('Must use valid cache')):
            second, ledger = polish_coherence(self.source, self.draft, self.config, self.cache, batch_size=2)
        self.assertEqual(first, second)
        self.assertTrue(all(row['cached'] for row in ledger))
        self.assertTrue(all(row['status'] == 'UNVERIFIED' for row in ledger))

    def test_one_invalid_response_retries_once_with_identical_inputs(self):
        (final, _), request = self.invoke(['{"1":"不完整。"}', '{"1":"回去吧。","2":"等一等。"}', '{"1":"好。"}'])
        self.assertEqual(final, self.draft)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[0].args, request.call_args_list[1].args)
        attempts = self.batch()['attempts']
        self.assertEqual(len(attempts), 2)
        self.assertIsNotNone(attempts[0]['error'])
        self.assertIsNone(attempts[1]['error'])

    def test_malformed_model_outputs_fail_after_two_calls_and_remain_recorded(self):
        malformed = ['{"1":"回去。","1":"别回去。","2":"等。"}',
                     '{"1":"回去。","3":"等。"}', '["回去。","等。"]',
                     '{"1":{"text":"回去。"},"2":"等。"}',
                     '{"1":"   ","2":"等。"}', '{"1":NaN,"2":"等。"}',
                     '```json\n{"1":"回去。","2":"等。"}\n```']
        for number, response in enumerate(malformed):
            with self.subTest(response=response):
                path = self.root/f'bad-{number}.json'
                with patch('src.coherence.call_llm', return_value=response) as request:
                    with self.assertRaises(CoherenceError):
                        polish_coherence(self.source, self.draft, self.config, path, batch_size=2)
                self.assertEqual(request.call_count, 2)
                run = next(iter(json.loads(path.read_text())['runs'].values()))
                attempts = next(iter(run['batches'].values()))['attempts']
                self.assertEqual([item['response'] for item in attempts], [response, response])

    def test_forbidden_output_text_is_rejected_without_srt_or_reasoning_leaks(self):
        invalid = ['\n\n新字幕', '回去。\n \n新的字幕', '00:00:02,000 --> 00:00:03,000',
                   '时间00:00:02.000', '回去\x00。', '\t回去。', '回去。\r', '回去\x7f。',
                   '\u202e回去。', '<think>内部思考</think>回去。', '</THINK>回去。',
                   '<analysis>思考</analysis>', '<|im_start|>assistant', '⟦E1⟧', '```', '字'*161]
        for number, text in enumerate(invalid):
            with self.subTest(text=repr(text)):
                response = json.dumps({'1': text, '2': '等。'})
                with patch('src.coherence.call_llm', return_value=response) as request:
                    with self.assertRaises(CoherenceError):
                        polish_coherence(self.source, self.draft, self.config, self.root/f'text-{number}.json', batch_size=2)
                self.assertEqual(request.call_count, 2)

    def test_later_failure_never_commits_partial_result_or_changes_input_files(self):
        output = self.root/'existing.zh.srt'
        output.write_bytes(b'existing subtitles remain unchanged')
        self.config.output_srt = output
        before = deepcopy((self.source, self.draft))
        with patch('src.coherence.call_llm', side_effect=['{"1":"回家吧。","2":"等等。"}', '{}', '{}']) as request:
            with self.assertRaises(CoherenceError):
                polish_coherence(self.source, self.draft, self.config, self.cache, batch_size=2)
        self.assertEqual(request.call_count, 3)
        self.assertEqual((self.source, self.draft), before)
        self.assertEqual(output.read_bytes(), b'existing subtitles remain unchanged')
        run = next(iter(self.saved()['runs'].values()))
        self.assertEqual(len(run['batches']), 2)
        self.assertEqual(sum('response' in entry for entry in run['batches'].values()), 1)

    def test_changed_context_model_sampler_draft_source_and_timing_invalidate_cache(self):
        changes = [lambda: self.context.write_text('不同的完整原始背景。'),
                   lambda: self.config.extra_payload.update(model='other-local-editor'),
                   lambda: self.config.extra_payload.update(seed=43),
                   lambda: setattr(self.draft[2], 'text', '好的。'),
                   lambda: setattr(self.source[2], 'text', 'はい、分かりました。'),
                   lambda: [setattr(rows[2], 'ts_line', '00:00:06,010 --> 00:00:07,000')
                            for rows in (self.source, self.draft)]]
        self.invoke()
        for count, change in enumerate(changes, 2):
            with self.subTest(change=count):
                change()
                (_, ledger), request = self.invoke()
                self.assertEqual(request.call_count, 2)
                self.assertTrue(all(not row['cached'] for row in ledger))
                self.assertEqual(len(self.saved()['runs']), count)

    def test_tampered_cache_response_requires_a_new_model_call(self):
        self.invoke()
        saved = self.saved()
        entry = next(iter(next(iter(saved['runs'].values()))['batches'].values()))
        entry['response'] = '{"1":"缓存被改动。","2":"等。"}'
        self.cache.write_text(json.dumps(saved, ensure_ascii=False))
        (_, ledger), request = self.invoke(['{"1":"回去吧。","2":"等一等。"}'])
        self.assertEqual(request.call_count, 1)
        self.assertEqual([row['cached'] for row in ledger], [False, False, True])
        self.assertTrue(self.batch()['cache_rejections'])

    def test_cached_json_is_revalidated_even_with_matching_response_hash(self):
        self.invoke()
        saved = self.saved()
        entry = next(iter(next(iter(saved['runs'].values()))['batches'].values()))
        response = '{"1":"旧。","1":"重复。","2":"等。"}'
        entry.update(response=response, response_sha256=hashlib.sha256(response.encode()).hexdigest(), cache_rejections='corrupt list')
        self.cache.write_text(json.dumps(saved, ensure_ascii=False))
        (_, ledger), request = self.invoke(['{"1":"回去吧。","2":"等一等。"}'])
        self.assertEqual(request.call_count, 1)
        self.assertFalse(ledger[0]['cached'])
        self.assertIn('Duplicate', self.batch()['cache_rejections'][0]['error'])

    def test_malformed_cache_bytes_are_preserved_and_do_not_approve_output(self):
        raw = b'{"version":"coherence-editor-1","runs":{},"runs":{}}'
        self.cache.write_bytes(raw)
        (_, ledger), request = self.invoke()
        self.assertEqual(request.call_count, 2)
        self.assertTrue(all(not row['cached'] for row in ledger))
        self.assertEqual(bytes.fromhex(self.saved()['previous_cache_hex']), raw)

    def test_input_pair_failures_stop_before_requests_and_cache_writes(self):
        variants = [(self.source, self.draft[:-1]),
                    (self.source, [replace(self.draft[0], index=2), *self.draft[1:]]),
                    (self.source, [replace(self.draft[0], ts_line='00:00:00,000 --> 00:00:01,000'), *self.draft[1:]]),
                    (self.source, [replace(self.draft[0], text=''), *self.draft[1:]]),
                    ([], []), ([{'index': 1}], self.draft)]
        for source, draft in variants:
            with self.subTest(source=source, draft=draft):
                with patch('src.coherence.call_llm') as request:
                    with self.assertRaises(ValueError):
                        polish_coherence(source, draft, self.config, self.cache)
                    request.assert_not_called()
                self.assertFalse(self.cache.exists())

    def test_only_explicit_local_editor_identity_and_bound_inputs_are_accepted(self):
        configs = [replace(self.config, endpoint='https://remote.invalid/v1/chat/completions'),
                   replace(self.config, endpoint='http://localhost.attacker.invalid/v1/chat/completions'),
                   replace(self.config, endpoint='http://user:secret@127.0.0.1:8089/v1/chat/completions'),
                   replace(self.config, target_lang='English')]
        for extra in ({}, {'model': ''}, {'model': 'gpt-6-astra'}, {'model': 'FABLE-5.1'},
                      {'model': 'local-editor', 'messages': []}, {'model': 'local-editor', 'max_tokens': True}):
            configs.append(replace(self.config, extra_payload=extra))
        for cfg in configs:
            with self.subTest(config=cfg):
                with patch('src.coherence.call_llm') as request:
                    with self.assertRaises(ValueError):
                        polish_coherence(self.source, self.draft, cfg, self.cache)
                    request.assert_not_called()
                self.assertFalse(self.cache.exists())

    def test_context_summary_only_used_without_original_files_and_limits_never_truncate(self):
        self.config = replace(self.config, context_files=[], context_summary='Complete supplied background')
        (_, _), request = self.invoke()
        self.assertIn('Complete supplied background', request.call_args_list[0].args[1])
        too_long = replace(self.config, context_summary='x'*(MAX_CONTEXT_CHARS+1))
        with patch('src.coherence.call_llm') as request:
            with self.assertRaisesRegex(ValueError, 'refusing truncation'):
                polish_coherence(self.source, self.draft, too_long, self.root/'too-long.json')
            long_draft = deepcopy(self.draft)
            long_draft[0].text = '字'*(MAX_DRAFT_CHARS+1)
            with self.assertRaisesRegex(ValueError, 'refusing truncation'):
                polish_coherence(self.source, long_draft, self.config, self.root/'too-long.json')
            request.assert_not_called()

    def test_context_cache_collision_and_unreadable_original_context_fail_before_calls(self):
        context_json = self.root/'original-context.json'
        context_json.write_text('{"original":"Background"}')
        collision_config = replace(self.config, context_files=[context_json])
        with patch('src.coherence.call_llm') as request:
            with self.assertRaisesRegex(ValueError, 'distinct coherence JSON'):
                polish_coherence(self.source, self.draft, collision_config, context_json)
            with patch('src.coherence.read_context_file', return_value=''):
                with self.assertRaisesRegex(ValueError, 'read completely'):
                    polish_coherence(self.source, self.draft, self.config, self.cache)
            request.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_transport_failure_is_bounded_and_recorded(self):
        with patch('src.coherence.call_llm', side_effect=RuntimeError('Local transport unavailable')) as request:
            with self.assertRaises(CoherenceError):
                polish_coherence(self.source, self.draft, self.config, self.cache)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(all('Local transport unavailable' in row['error'] for row in self.batch()['attempts']))



class MessageLayoutTests(unittest.TestCase):
    def test_default_payload_remains_one_user_message(self):
        from src.translate import _build_payload
        cfg = TranslateConfig(extra_payload={'temperature': .3})
        payload = _build_payload('batch data', 'stable instruction', cfg, {}, 512, stream=False)
        self.assertEqual(payload['messages'], [{'role': 'user', 'content': 'stable instruction\nbatch data'}])
        self.assertEqual(payload['max_tokens'], 512)

    def test_opt_in_creates_real_boundary_without_changing_other_settings(self):
        from src.translate import _build_payload
        cfg = TranslateConfig(extra_payload={'temperature': .3}, telemetry_path=Path('metrics.jsonl'))
        old = _build_payload('batch data', 'stable instruction', cfg, {'reasoning_effort':'none'}, 512, stream=True)
        new = _build_payload('batch data', 'stable instruction', replace(cfg, separate_instruction=True), {'reasoning_effort':'none'}, 512, stream=True)
        self.assertEqual(new.pop('messages'), [{'role':'system','content':'stable instruction'}, {'role':'user','content':'batch data'}])
        old.pop('messages')
        self.assertEqual(new, old)

    def test_non_boolean_layout_is_rejected(self):
        for value in ('true', 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TranslateConfig(separate_instruction=value)

if __name__ == '__main__':
    unittest.main()
