"""Offline checks for the provisional selected pipeline; no model or media calls."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src import selected_asr as asr
from src import selected_pipeline as pipeline
from src.translate import SrtBlock


class SelectedPipelineChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.recipe = {'model':pipeline.MODEL,'sha256':pipeline.WEIGHT_SHA256,
            'sampler':dict(pipeline.SAMPLER),'context_size':32768,'cpu_moe_layers':0,
            'threads':4,'full_swa':False,'restore_model':'qwen3.8-27b-dflash',
            'admin_url':'http://127.0.0.1:8089/admin','weights':'synthetic.gguf',
            'server_binary':'synthetic-server'}

    def test_window_geometry_complete_and_native_bounded(self):
        for duration in (0.1, 2, 25, 26, 28, 30, 31, 40, 47, 77):
            pcm = np.zeros(round(duration*16000), dtype='<i2').tobytes()
            windows = asr.partition_pcm(pcm)
            self.assertEqual(windows[0]['start_frame'],0)
            self.assertEqual(windows[-1]['end_frame'],len(pcm)//2)
            self.assertEqual([w['number'] for w in windows],list(range(1,len(windows)+1)))
            for i,w in enumerate(windows):
                self.assertLessEqual(w['end_frame']-w['start_frame'],25*16000)
                self.assertGreater(w['end_frame'],w['start_frame'])
                if i:self.assertEqual(w['start_frame'],windows[i-1]['end_frame'])

    def test_supported_historical_geometry_unchanged(self):
        rng = np.random.default_rng(1234)
        for duration in (1, 18, 25, 40, 60, 120):
            pcm = rng.integers(-500,500,size=duration*16000,dtype=np.int16).tobytes()
            old = asr._historical_partition_pcm(pcm)
            if old[-1]['end_frame']-old[-1]['start_frame']<=25*16000:
                self.assertEqual(asr.partition_pcm(pcm),old)

    def test_pinned_generation_policy(self):
        first=asr.generation_policy(1,20260915)
        self.assertEqual(first['max_new_tokens'],440)
        self.assertEqual(first['language'],'ja'); self.assertEqual(first['task'],'transcribe')
        self.assertFalse(first['do_sample']);self.assertFalse(first['return_timestamps'])
        self.assertFalse(first['condition_on_prev_tokens'])
        self.assertEqual(asr.generation_policy(2,20260915)['temperature'],.3)
        self.assertEqual(asr.seed_for(9,3),20460923)

    def test_complete_token_sequence_required(self):
        decoder={'prefix_ids':[10,11,12,13],'eos_token_id':2,'pad_token_id':2,'vocab_size':100}
        self.assertEqual(asr.token_completion([10,11,12,13,21,2,2],decoder)['generated_tokens_through_eos'],2)
        for values in ([10,11,12,13,21], [10,11,12,13,2,21], [10,11,12,13,True,2]):
            with self.assertRaises(ValueError):asr.token_completion(values,decoder)

    def test_source_empty_window_stays_in_coverage_ledger(self):
        windows=[{'number':1,'start_frame':0,'end_frame':16000,'sha256':'a'},
                 {'number':2,'start_frame':16000,'end_frame':32000,'sha256':'b'}]
        rows=[]
        for w,text in zip(windows,['','テスト']):
            rows.append({'window':w['number'],'start_frame':w['start_frame'],'end_frame':w['end_frame'],
                         'audio_sha256':w['sha256'],'text':text,'accepted_text_sha256':asr.text_hash(text),
                         'repetition_check':asr.compression_check(text,asr.REPETITION_POLICY)})
        source,ledger=asr.assemble_windows({'windows':windows},rows,repetition_guard=asr.REPETITION_POLICY)
        self.assertEqual(len(source),1);self.assertEqual(len(ledger),2)
        self.assertIsNone(ledger[0]['source_index']);self.assertEqual(source[0].index,1)
        self.assertFalse(ledger[0]['silence_verified'])

    def test_translation_request_contract(self):
        source=[SrtBlock(1,'00:00:00,000 --> 00:00:02,000','テスト')]
        body,instruction,schema=pipeline._translation_inputs(source,'synthetic context\n')
        self.assertEqual(json.loads(body),{'1':'テスト'})
        self.assertEqual(instruction,pipeline.INSTRUCTION+'\n完整原始作品背景（只读）：\nsynthetic context\n')
        self.assertEqual(schema['required'],['1']);self.assertFalse(schema['additionalProperties'])
        cfg=pipeline.configuration('http://127.0.0.1:18101/v1/chat/completions',self.recipe)
        self.assertEqual(cfg.retries,0);self.assertEqual(cfg.max_tokens,16384)
        self.assertEqual(cfg.extra_payload['reasoning_budget_tokens'],1536)
        self.assertEqual(cfg.extra_payload['chat_template_kwargs'],{'enable_thinking':True})
        self.assertEqual({k:cfg.extra_payload[k] for k in pipeline.SAMPLER},pipeline.SAMPLER)

    def test_response_all_ids_and_geometry_preserved(self):
        source=[SrtBlock(1,'00:00:00,000 --> 00:00:02,000','テスト'),
                SrtBlock(2,'00:00:02,000 --> 00:00:04,000','テスト')]
        target,_=pipeline.parse_translation('{"1":"测试","2":"测试二"}',source)
        self.assertEqual([r.ts_line for r in target],[r.ts_line for r in source])
        for raw in ('{"1":"测试"}', '{"1":"测试","1":"重复","2":"测试"}', '{"1":"","2":"测试"}'):
            with self.assertRaises(ValueError):pipeline.parse_translation(raw,source)

    def test_native_capacity_fails_without_truncation(self):
        cfg=pipeline.configuration('http://127.0.0.1:18101/v1/chat/completions',self.recipe)
        responses=[{'prompt':'synthetic'}, {'tokens':list(range(17000))}]
        with patch.object(pipeline,'_request_json',side_effect=responses):
            result=pipeline.capacity_check('body','instruction',cfg,{'default_generation_settings':{'n_ctx':32768}})
        self.assertFalse(result['fits']);self.assertEqual(result['prompt_tokens'],17000)
        self.assertTrue(result['full_context_preserved'])

    def _asr_preparation(self):
        identity={'model_dir':str(self.folder/'synthetic-model')}
        return {'asr_identity':identity,'recipe':self.recipe}

    def test_asr_timeout_stops_then_restores(self):
        events=[]
        class Process:
            def wait(self,timeout):raise subprocess.TimeoutExpired('synthetic',timeout)
        def request(url,payload=None,**kwargs):
            events.append('status' if url.endswith('/status') else 'restore' if url.endswith('/load') else 'unload')
            return {'loaded':'qwen3.8-27b-dflash','loaded_model':'qwen3.8-27b-dflash'}
        with patch.object(asr,'validate_spec'),patch.object(pipeline.subprocess,'Popen',return_value=Process()),\
             patch.object(pipeline,'_request_json',side_effect=request),\
             patch.object(pipeline,'_stop_owned_process',side_effect=lambda process:events.append('stop')):
            with self.assertRaises(subprocess.TimeoutExpired):
                pipeline._run_asr({},self._asr_preparation(),self.folder,timeout=1)
        self.assertEqual(events,['status','unload','stop','restore'])
        receipt=json.loads((self.folder/'asr/lifecycle.json').read_text())
        self.assertTrue(receipt['original_backend_restored'])

    def test_unconfirmed_worker_stop_blocks_restore(self):
        events=[]
        class Process:
            def wait(self,timeout):raise subprocess.TimeoutExpired('synthetic',timeout)
        with patch.object(asr,'validate_spec'),patch.object(pipeline.subprocess,'Popen',return_value=Process()),\
             patch.object(pipeline,'_request_json',side_effect=lambda url,*a,**k:events.append('status' if url.endswith('/status') else 'unload') or {'loaded_model':'qwen3.8-27b-dflash'}),\
             patch.object(pipeline,'_stop_owned_process',side_effect=RuntimeError('synthetic shutdown failure')):
            with self.assertRaises(subprocess.TimeoutExpired):
                pipeline._run_asr({},self._asr_preparation(),self.folder,timeout=1)
        self.assertEqual(events,['status','unload'])
        receipt=json.loads((self.folder/'asr/lifecycle.json').read_text())
        self.assertFalse(receipt['owned_worker_stopped'])
        self.assertIsNone(receipt['original_backend_restored'])

    def test_wrong_active_backend_is_not_unloaded(self):
        with patch.object(asr,'validate_spec'), patch.object(pipeline,'_request_json',return_value={'loaded_model':'different'}) as http, patch.object(pipeline.subprocess,'Popen') as process:
            with self.assertRaisesRegex(RuntimeError,'not confirmed active'):
                pipeline._run_asr({},self._asr_preparation(),self.folder,timeout=1)
        self.assertEqual(http.call_count,1);process.assert_not_called()
        receipt=json.loads((self.folder/'asr/lifecycle.json').read_text())
        self.assertIsNone(receipt['original_backend_restored'])

    def test_writer_length_stop_cannot_commit_and_restores(self):
        source=[SrtBlock(1,'00:00:00,000 --> 00:00:02,000','テスト')]
        context=self.folder/'context.txt';context.write_text('synthetic')
        prep={'recipe':self.recipe,'context_file':str(context),'request_timeout_seconds':600}
        events=[]
        @contextmanager
        def backend(*args,**kwargs):
            events.append('enter')
            try:yield ('http://127.0.0.1:18101/v1/chat/completions',{})
            finally:events.append('restore')
        def generate(body,instruction,cfg,**kwargs):
            raw='{"1":"测试"}'
            (self.folder/'writer-metrics.jsonl').write_text(json.dumps({'stage':cfg.stage,'thinking':True,'finish_reason':'length','answer_chars':len(raw)})+'\n')
            return raw
        with patch.object(pipeline,'_confirm_original_backend'),patch.object(pipeline,'temporary_local_writer',side_effect=backend),patch.object(pipeline,'capacity_check',return_value={'fits':True}),patch.object(pipeline,'call_llm',side_effect=generate):
            with self.assertRaisesRegex(RuntimeError,'normal-stop'):
                pipeline._translate(source,prep,self.folder)
        self.assertEqual(events,['enter','restore'])
        self.assertFalse((self.folder/'translation.json').exists())
        receipt=json.loads((self.folder/'writer-lifecycle.json').read_text())
        self.assertTrue(receipt['original_backend_restored']);self.assertFalse(receipt['semantic_artifact_preserved'])

    def test_clean_backend_recipe_has_no_refinement_stages(self):
        weight=self.folder/'model.gguf';weight.write_bytes(b'synthetic')
        binary=self.folder/'server';binary.write_bytes(b'synthetic');binary.chmod(0o700)
        recipe={'version':1,'model':pipeline.MODEL,'weights':weight.name,'sha256':pipeline.WEIGHT_SHA256,
                'server_binary':binary.name,'sampler':dict(pipeline.SAMPLER)}
        path=self.folder/'backend.json';path.write_text(json.dumps(recipe))
        with patch.object(pipeline,'_weight_bundle') as verify:
            loaded=pipeline.load_writer_recipe(path)
        self.assertEqual(loaded['context_size'],32768)
        self.assertEqual(loaded['weights'],str(weight));self.assertNotIn('stages',loaded)
        verify.assert_called_once_with(weight,pipeline.WEIGHT_SHA256,None)

    def test_clean_backend_recipe_rejects_boolean_sampler(self):
        recipe={'version':1,'model':pipeline.MODEL,'weights':'unused','sha256':pipeline.WEIGHT_SHA256,
                'server_binary':'unused','sampler':dict(pipeline.SAMPLER,temperature=True)}
        path=self.folder/'backend.json';path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError,'sampler'):pipeline.load_writer_recipe(path)

    def test_preexisting_output_rejected_before_model_validation(self):
        media=self.folder/'media.wav';media.write_bytes(b'fake')
        context=self.folder/'context.txt';context.write_text('synthetic')
        recipe=self.folder/'recipe.json';recipe.write_text('{}')
        config=pipeline.PipelineConfig(media,self.folder,context,self.folder/'model',recipe)
        with patch.object(asr,'bundle_identity') as native:
            with self.assertRaisesRegex(ValueError,'already exists'):pipeline.prepare(config)
        native.assert_not_called()

    def test_display_uses_existing_model_formatter_and_exact_text_guard(self):
        source=[SrtBlock(1,'00:00:00,000 --> 00:00:02,000','テスト')]
        target=[SrtBlock(1,source[0].ts_line,'测试')]
        def formatter(a,b,metadata,config,cache,**kwargs):
            self.assertEqual(config.retries,2);self.assertEqual(config.max_tokens,2048)
            self.assertTrue(config.pause_layout);self.assertTrue(kwargs['model_failure_fallback'])
            self.assertEqual(metadata['utterance_tokens'],[[]])
            return a,b,[{'line':1,'utterance':1}]
        with patch.object(pipeline,'build_display',side_effect=formatter):
            self.assertEqual(pipeline._display(source,target,self.folder)['display_cues'],1)
        with patch.object(pipeline,'build_display',return_value=(source,[SrtBlock(1,source[0].ts_line,'改变')],[])):
            with self.assertRaisesRegex(RuntimeError,'changed semantic text'):
                pipeline._display(source,target,self.folder)


if __name__=='__main__':unittest.main()
