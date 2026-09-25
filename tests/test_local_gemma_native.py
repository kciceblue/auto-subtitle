"""Generic local request transport tested with synthetic SSE and capacity RPCs."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import requests
from src import local_gemma_native as native
from src import translate


class FakeResponse:
    def __init__(self, answer, *, reasoning='', finish='stop', done=True,
                 usage=None, fail_after=False, status=200):
        self.status_code = status
        self.closed = False
        self.fail_after = fail_after
        usage = {'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300} if usage is None else usage
        events = [
            {'choices': [{'index': 0, 'delta': {'reasoning_content': reasoning}}]},
            {'choices': [{'index': 0, 'delta': {'content': answer}, 'finish_reason': finish}]},
            {'choices': [], 'usage': usage, 'timings': {'predicted_n': 200}},
        ]
        self.lines = [('data: '+json.dumps(event, ensure_ascii=False)).encode() for event in events]
        if done: self.lines.append(b'data: [DONE]')

    def iter_lines(self, decode_unicode=False):
        for index, line in enumerate(self.lines):
            yield line.decode() if decode_unicode else line
            if self.fail_after and index == 1: raise requests.ConnectionError('synthetic stream loss')

    def raise_for_status(self):
        if self.status_code != 200: raise requests.HTTPError('synthetic HTTP failure')

    def close(self): self.closed = True


class LocalGemmaNativeFixture(unittest.TestCase):
    """No models: arbitrary sparse IDs exercise the generic contract."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.result = {'owners': {'2': '  合成文本。\n', '9': 'literal name'}}
        self.request = {'instruction': 'Synthetic fixed local instruction',
            'body': {'owners': [{'id': 2, 'source': 'synthetic two'}, {'id': 9, 'source': 'synthetic nine'}]},
            'schema': {'type': 'object', 'additionalProperties': False, 'required': ['owners'], 'properties': {
                'owners': {'type': 'object', 'additionalProperties': False, 'required': ['2', '9'],
                           'properties': {'2': {'type': 'string'}, '9': {'type': 'string'}}}}}}
        self.settings = native.GemmaSettings()
        self.prepared = self.prepare()
        self.template = 'synthetic exact native template'
        self.identity = {'model_alias': self.settings.model, 'model_sha256': self.settings.model_sha256,
                         'model_path': '/synthetic/gemma.gguf', 'pid': 123, 'context_size': self.settings.context_size,
                         'full_swa': False, 'chat_template_sha256': native.digest(self.template.encode())}
        self.calls = []

    def prepare(self, settings=None):
        return native.prepare_request(self.request, settings or self.settings, contract_id='synthetic-v1')

    def rpc(self, endpoint, payload=None, timeout=20):
        self.calls.append((endpoint, deepcopy(payload)))
        if endpoint.endswith('/props'):
            return {'model_alias': self.identity['model_alias'], 'model_path': self.identity['model_path'],
                    'chat_template': self.template, 'default_generation_settings': {'n_ctx': self.identity['context_size']}}
        if endpoint.endswith('/apply-template'): return {'prompt': 'synthetic rendered full request'}
        if endpoint.endswith('/tokenize'): return {'tokens': list(range(100))}
        self.fail('Unexpected capacity endpoint')

    def invoke(self, key='draft', response=None, prepared=None, rpc=None, deadline=None):
        response = response or FakeResponse(json.dumps(self.result, ensure_ascii=False))
        with patch.object(native, '_request_json', side_effect=rpc or self.rpc), \
             patch.object(native.requests, 'post', return_value=response) as post:
            value = native.ask(self.directory, key, prepared or self.prepared, self.identity,
                               time.monotonic()+60 if deadline is None else deadline)
        return value, post

    def saved(self, key='draft'): return native.strict_json((self.directory/(key+'.json')).read_bytes())

    def reseal(self, saved, key='draft'):
        saved['receipt_sha256'] = native.fingerprint({k:v for k,v in saved.items() if k != 'receipt_sha256'})
        (self.directory/(key+'.json')).write_text(json.dumps(saved, ensure_ascii=False))


class LocalGemmaNativeTests(LocalGemmaNativeFixture):
    def test_one_real_stream_exact_payload_and_network_free_cache_replay(self):
        original = deepcopy(self.request); response = FakeResponse(json.dumps(self.result, ensure_ascii=False))
        result, post = self.invoke(response=response)
        self.assertEqual(result, self.result); self.assertEqual(self.request, original)
        self.assertEqual(post.call_count, 1); self.assertTrue(response.closed)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload, self.prepared['payload']); self.assertEqual(self.calls[1][1], payload)
        self.assertEqual(payload['stream_options'], {'include_usage': True})
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual((payload['reasoning_effort'], payload['reasoning_budget_tokens']), ('none', 0))
        self.assertEqual(payload['max_tokens'], 16384)
        self.assertFalse(post.call_args.kwargs['allow_redirects'])
        self.assertTrue(all(url.startswith(native.CAPACITY_BASE+'/') for url, _ in self.calls))
        saved = self.saved(); self.assertEqual(saved['attempt']['capacity']['input_ids'], list(range(100)))
        self.assertEqual(saved['attempt']['generation_http_calls'], 1)
        self.assertEqual(saved['response']['reasoning'], '')
        self.assertEqual(saved['parsed']['owners']['2'], '  合成文本。\n')
        with patch.object(native, '_request_json', side_effect=AssertionError('no RPC')), \
             patch.object(native.requests, 'post', side_effect=AssertionError('no generation')):
            self.assertEqual(native.replay_native(self.directory/'draft.json', self.prepared, self.identity), self.result)
            self.assertEqual(native.ask(self.directory, 'draft', self.prepared, self.identity, time.monotonic()+60), self.result)

    def test_stream_framing_above_four_megabytes_does_not_reduce_token_allowance(self):
        response = FakeResponse(json.dumps(self.result))
        # Valid per-event metadata can exceed the former cap with fewer than 16K deltas.
        event = b'data: '+json.dumps({'id':'x'*230, 'choices':[{'index':0,'delta':{}}]}).encode()
        response.lines[0:0] = [event]*14000
        self.assertGreater(sum(len(line)+1 for line in response.lines), 4_000_000)
        result, post = self.invoke('framing', response)
        self.assertEqual(result, self.result); self.assertEqual(post.call_count, 1)
        self.assertEqual(self.saved('framing')['response']['usage']['completion_tokens'], 200)
        self.assertLess((self.directory/'framing.sse').stat().st_size, native.MAX_STREAM_BYTES)

    def test_positive_reasoning_budgets_preserved_without_effort_override(self):
        for budget in (1536, 4096, 8192):
            with self.subTest(budget=budget):
                prepared = self.prepare(replace(self.settings, thinking=True, reasoning_budget_tokens=budget, maximum_seconds=3180))
                payload = prepared['payload']; self.assertNotIn('reasoning_effort', payload)
                self.assertEqual(payload['reasoning_budget_tokens'], budget)
                self.assertEqual(payload['max_tokens'], 16384)
                self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': True})
                reasoning = '  Synthetic private reasoning.\n'
                result, post = self.invoke(str(budget), FakeResponse(json.dumps(self.result), reasoning=reasoning), prepared)
                self.assertEqual(result, self.result); self.assertEqual(post.call_count, 1)
                self.assertEqual(self.saved(str(budget))['response']['reasoning'], reasoning)
                self.assertIn(reasoning.encode().replace(b'\n', b'\\n'), (self.directory/(str(budget)+'.sse')).read_bytes())
        for values in ({'thinking':True, 'reasoning_budget_tokens':0}, {'thinking':True, 'reasoning_budget_tokens':16385},
                       {'thinking':False, 'reasoning_budget_tokens':1}, {'reasoning_budget_tokens':True}):
            with self.assertRaises(ValueError): replace(self.settings, **values)

    def test_stream_projection_agrees_with_shared_parser_and_keeps_reasoning(self):
        response = FakeResponse(json.dumps(self.result), reasoning='synthetic reasoning')
        projected = native.decode_stream(b'\n'.join(response.lines)+b'\n')
        with patch('src.translate.requests.post', return_value=response) as post:
            previous = translate._stream_response(native.ENDPOINT, self.prepared['payload'], 60, 1000)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(projected['content'], previous.content)
        self.assertEqual(len(projected['reasoning']), previous.reasoning_chars)
        self.assertEqual(projected['finish_reason'], previous.finish_reason)
        self.assertEqual(projected['usage'], previous.usage)

    def test_reasoning_adherence_failure_preserves_valid_answer_and_never_retries(self):
        for key, thinking, reasoning in (('missing', True, ''), ('unexpected', False, 'unexpected reasoning')):
            prepared = self.prepare(replace(self.settings, thinking=thinking, reasoning_budget_tokens=4096 if thinking else 0))
            response = FakeResponse(json.dumps(self.result), reasoning=reasoning)
            with patch.object(native, '_request_json', side_effect=self.rpc), patch.object(native.requests, 'post', return_value=response) as post:
                with self.assertRaises(native.TreatmentAdherenceError):
                    native.ask(self.directory, key, prepared, self.identity, time.monotonic()+60)
                self.assertEqual(post.call_count, 1)
            saved = self.saved(key); self.assertEqual(saved['status'], 'failed')
            self.assertEqual(json.loads(saved['response']['content']), self.result)
            self.assertEqual(saved['attempt']['error_type'], 'TreatmentAdherenceError')
            self.assertTrue(response.closed)
            with patch.object(native.requests, 'post', side_effect=AssertionError('no reroll')):
                with self.assertRaises(ValueError): native.ask(self.directory, key, prepared, self.identity, time.monotonic()+60)

    def test_malformed_duplicate_missing_owner_length_and_incomplete_are_terminal(self):
        cases = [FakeResponse('{'), FakeResponse('{"owners":{},"owners":{}}'), FakeResponse('{"owners":{"2":"x"}}'),
                 FakeResponse(json.dumps(self.result), finish='length'), FakeResponse(json.dumps(self.result), done=False),
                 FakeResponse(json.dumps(self.result), usage={})]
        for index, response in enumerate(cases):
            key = f'bad-{index}'
            with self.subTest(index=index), patch.object(native, '_request_json', side_effect=self.rpc), \
                 patch.object(native.requests, 'post', return_value=response) as post:
                with self.assertRaises(Exception): native.ask(self.directory, key, self.prepared, self.identity, time.monotonic()+60)
                self.assertEqual(post.call_count, 1)
            self.assertEqual(self.saved(key)['status'], 'failed'); self.assertTrue(response.closed)
            self.assertTrue((self.directory/(key+'.sse')).is_file())
            with patch.object(native.requests, 'post', side_effect=AssertionError('no reroll')):
                with self.assertRaises(ValueError): native.ask(self.directory, key, self.prepared, self.identity, time.monotonic()+60)

    def test_transport_loss_and_http_failure_are_closed_single_calls(self):
        for key, response in (('loss', FakeResponse('{', reasoning='partial reasoning', fail_after=True)),
                              ('http', FakeResponse('{}', status=503))):
            with patch.object(native, '_request_json', side_effect=self.rpc), patch.object(native.requests, 'post', return_value=response) as post:
                with self.assertRaises((ValueError, requests.ConnectionError)):
                    native.ask(self.directory, key, self.prepared, self.identity, time.monotonic()+60)
                self.assertEqual(post.call_count, 1)
            self.assertTrue(response.closed); self.assertEqual(self.saved(key)['status'], 'failed')
        self.assertEqual(self.saved('loss')['response']['content'], '{')
        self.assertEqual(self.saved('loss')['response']['reasoning'], 'partial reasoning')
        self.assertFalse(self.saved('loss')['response']['done'])
        self.assertEqual(self.saved('http')['attempt']['http_status'], 503)

    def test_actual_context_template_and_full_capacity_reject_before_generation(self):
        for name in ('context', 'template', 'overflow'):
            def rpc(url, payload=None, timeout=20):
                value = self.rpc(url, payload, timeout)
                if url.endswith('/props') and name == 'context': value['default_generation_settings']['n_ctx'] = 32768
                if url.endswith('/props') and name == 'template': value['chat_template'] = 'changed template'
                if url.endswith('/tokenize') and name == 'overflow': value = {'tokens': [1]*(163840-16384-64+1)}
                return value
            with self.subTest(name=name), patch.object(native, '_request_json', side_effect=rpc), patch.object(native.requests, 'post') as post:
                with self.assertRaises(ValueError): native.ask(self.directory, name, self.prepared, self.identity, time.monotonic()+60)
                self.assertEqual(post.call_count, 0)
            self.assertEqual(self.saved(name)['attempt']['generation_http_calls'], 0)
        self.assertFalse(self.saved('overflow')['attempt']['capacity']['fits'])

    def test_expected_request_and_saved_stream_tampering_rejects(self):
        self.invoke(); original = self.saved()
        for name in ('request', 'settings', 'template', 'parsed'):
            saved = deepcopy(original)
            if name == 'request': saved['prepared']['request']['body']['owners'][0]['source'] = 'different'
            if name == 'settings': saved['prepared']['settings']['reasoning_budget_tokens'] = 4096
            if name == 'template': saved['attempt']['capacity']['props']['chat_template'] = 'different'
            if name == 'parsed': saved['parsed']['owners']['2'] = 'different'
            self.reseal(saved)
            with self.subTest(name=name), self.assertRaises(ValueError):
                native.replay_native(self.directory/'draft.json', self.prepared, self.identity)
        self.reseal(original); path = self.directory/'draft.sse'; path.write_bytes(path.read_bytes()+b'data: {}\n')
        with self.assertRaises(ValueError): native.replay_native(self.directory/'draft.json', self.prepared, self.identity)

    def test_contract_validation_is_explicit_and_cannot_mutate_input(self):
        called = []
        def validator(request): called.append(deepcopy(request))
        self.assertEqual(native.prepare_request(self.request, self.settings, contract_id='test', validate_request=validator)['request'], self.request)
        self.assertEqual(called, [self.request])
        original = deepcopy(self.request)
        def mutate(request): request['body']['owners'].clear()
        with self.assertRaisesRegex(ValueError, 'mutated'):
            native.prepare_request(self.request, self.settings, contract_id='test', validate_request=mutate)
        self.assertEqual(self.request, original)
        changed = deepcopy(self.prepared); changed['payload']['reasoning_effort'] = 'high'
        with patch.object(native.requests, 'post') as post:
            with self.assertRaises(ValueError): native.ask(self.directory, 'changed', changed, self.identity, time.monotonic()+60)
            self.assertEqual(post.call_count, 0)

    def test_deadline_failure_allows_caller_cleanup_and_cannot_resume(self):
        cleaned = []
        with patch.object(native, '_request_json', side_effect=TimeoutError('synthetic expired capacity')), \
             patch.object(native.requests, 'post') as post:
            try:
                with self.assertRaises(TimeoutError): native.ask(self.directory, 'deadline', self.prepared, self.identity, time.monotonic()+60)
            finally: cleaned.append(True)
            self.assertEqual(post.call_count, 0)
        self.assertEqual(cleaned, [True]); self.assertEqual(self.saved('deadline')['status'], 'failed')
        with patch.object(native.requests, 'post') as post:
            with self.assertRaises(ValueError): native.ask(self.directory, 'before', self.prepared, self.identity, time.monotonic()-1)
            self.assertEqual(post.call_count, 0)
        (self.directory/'orphan.sse').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            native.ask(self.directory, 'orphan', self.prepared, self.identity, time.monotonic()+60)


if __name__ == '__main__': unittest.main()
