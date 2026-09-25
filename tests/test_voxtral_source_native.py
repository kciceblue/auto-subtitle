"""Actual Voxtral frontend and tiny randomly initialized architecture, CPU only.

No checkpoint weights, production inputs, CUDA or backend calls are used.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor,VoxtralRealtimeForConditionalGeneration,GenerationConfig
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import VoxtralRealtimeConfig,VoxtralRealtimeEncoderConfig,VoxtralRealtimeTextConfig
from src import voxtral_source_native as n

MODEL_DIR=Path(__file__).resolve().parents[1]/'models/voxtral-mini-4b-realtime-2602'


def timestamp(ms):
    h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def tiny_model(processor,mode='duration'):
    config=VoxtralRealtimeConfig(audio_config=VoxtralRealtimeEncoderConfig(hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,head_dim=8,num_mel_bins=128,sliding_window=750),
        text_config=VoxtralRealtimeTextConfig(vocab_size=131072,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            sliding_window=8192,max_position_embeddings=8192,pad_token_id=11,bos_token_id=1,eos_token_id=2),
        hidden_size=16)
    config._attn_implementation='sdpa'
    model=VoxtralRealtimeForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    model.generation_config=GenerationConfig.from_pretrained(MODEL_DIR,local_files_only=True)
    emitted=[]
    def deterministic(module,args,kwargs,out):
        # The actual native encoder/projector/decoder/cache still execute.
        # Only final logits are synthetic, controlling the stop experiment.
        pos=int(kwargs['cache_position'][-1]);token=32
        if mode=='early_eos':token=2
        if mode=='late_eos' and pos>=60:token=2
        if mode=='lexical' and not emitted:token=processor.tokenizer.encode('a',add_special_tokens=False)[0]
        out.logits.fill_(-1000);out.logits[...,token]=1000;emitted.append(pos)
        return out
    model.language_model.register_forward_hook(deterministic,with_kwargs=True)
    return model,emitted


class VoxtralFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor=AutoProcessor.from_pretrained(MODEL_DIR,local_files_only=True)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.audio=np.zeros(32000,dtype=np.float32);self.audio[-320:]=np.float32(.012345)
        self.rows=[{'index':i,'ts_line':timestamp((i-1)*2000)+' --> '+timestamp(i*2000),'text':'invented'} for i in range(1,67)]
        self.detector={'mono':{'frames':66*32000,'sample_rate':16000},'vad_speech':[{'start':1,'end':12}]}
        self.bindings={}
        for slot in ['silence','noise']+[f'owner-{i:03d}' for i in range(1,67)]:
            audio=np.zeros(192000,dtype=np.float32) if slot in ('silence','noise') else self.audio
            p=self.folder/(str(len(audio))+'.wav')
            if not p.exists():sf.write(p,audio,16000,subtype='FLOAT')
            self.bindings[slot]={'path':str(p),'sha256':n._file(p),'pcm_sha256':n._sha(audio.tobytes()),'frames':len(audio),'sample_rate':16000,'dtype':'float32'}
        self.assets={'invented_assets':True};self.identity={'family':'voxtral_realtime','worker_sha256':n._file(n.__file__)}
        mocks=[patch.object(n,'execution_identity',return_value=self.identity),patch.object(n,'validate_assets',return_value=self.identity),
            patch.object(n,'model_config',return_value={'generation_config':n._read(MODEL_DIR/'generation_config.json')}),patch.object(torch.cuda,'manual_seed_all'),
            patch.object(n,'_validate_runtime',side_effect=self.cpu_runtime)]
        for mock in mocks:mock.start();self.addCleanup(mock.stop)
        self.plan=n.build_plan(self.rows,self.detector,'a'*64,self.bindings,self.assets,'b'*64)
        self.request=n.build_request(self.plan,'owner-001');self.capture=None
    @staticmethod
    def cpu_runtime(value):
        if value['device']!='cpu' or value['dtype']!='torch.bfloat16' or value['audio_layers']!=2 or value['text_layers']!=2:
            raise ValueError('Synthetic CPU model changed')
        return value
    def call(self,mode='duration',*,deadline=None):
        model,emitted=tiny_model(self.processor,mode)
        def save(value):self.capture=value
        n.capture_call(model,self.processor,self.audio,self.request,artifact_root=self.folder,save_capture=save,
            deadline=time.monotonic()+30 if deadline is None else deadline)
        return model,emitted
    def receipt(self):
        c=self.capture
        value={'version':n.VERSION,'request':deepcopy(self.request),'request_sha256':n.ec._hash(self.request),
            'assets_sha256':n.ec._hash(self.assets),'started_utc':'2026-01-01T00:00:00+00:00','finished_utc':n._now(),
            'seconds':1.,'status':'empty' if c['parsed_text']=='' else 'complete','error_type':None,'capture':deepcopy(c)}
        value['receipt_sha256']=n.ec._hash(value);return value
    def replay(self,value):
        return n.validate_native_receipt(value,self.request,self.plan,assets=self.assets,artifact_root=self.folder,processor=self.processor)


class VoxtralSourceNativeTests(VoxtralFixture):
    def test_actual_whole_frontend_padding_pcm16_and_short_long_clock(self):
        for frames in (192000,400000,1920000):
            audio=np.zeros(frames,dtype=np.float32);audio[-320:]=np.float32(.012345)
            inputs,layout=n._frontend(self.processor,audio)
            self.assertEqual(layout['left_pad_samples'],40960)
            self.assertEqual(layout['prefix_positions'],39)
            self.assertEqual(layout['native_total_positions'],(frames+1279)//1280+49)
            self.assertEqual(inputs['input_features'].shape[-1],layout['native_total_positions']*8)
            serialized=layout['serialized_pcm'];self.assertEqual(len(serialized),frames)
            self.assertFalse(np.array_equal(serialized,audio))
            self.assertLessEqual(float(np.max(np.abs(serialized-audio))),1/32768)
            self.assertTrue(np.array_equal(layout['padded_pcm'][40960:40960+frames],serialized))
        self.assertGreater(layout['native_total_positions'],512)

    def test_actual_tiny_encoder_generation_slices_and_duration_completion_replay(self):
        model,emitted=self.call();c=self.capture
        self.assertEqual(c['coverage']['stop_reason'],'audio_duration')
        self.assertTrue(c['coverage']['physical_audio_complete'])
        self.assertNotEqual(c['output_token_ids'][-1],2)
        self.assertEqual(c['coverage']['output_positions'],c['layout']['native_total_positions'])
        self.assertEqual(c['audio_calls'][0]['position_end'],39)
        self.assertTrue(all(x['encoder_completed'] and x['layer_calls']==2 for x in c['audio_calls']))
        self.assertEqual(c['embedder_calls'],1);self.assertEqual(c['generation_calls'],1)
        self.assertEqual(self.replay(self.receipt())['parsed_text'],'')
        self.assertNotIn('_prepare_generated_length',model.__dict__)
        self.assertFalse(model._forward_pre_hooks);self.assertFalse(model.audio_tower._forward_hooks)

    def test_same_owned_tiny_model_resets_native_cache_between_calls(self):
        model,emitted=tiny_model(self.processor)
        captures=[]
        for _ in range(2):
            n.capture_call(model,self.processor,self.audio,self.request,artifact_root=self.folder,
                save_capture=lambda value:setattr(self,'capture',value),deadline=time.monotonic()+30)
            captures.append(deepcopy(self.capture))
        self.assertEqual(captures[0]['output_token_ids'],captures[1]['output_token_ids'])
        for capture in captures:
            self.assertEqual(capture['audio_calls'][0]['position_start'],0)
            self.assertEqual(capture['generation_calls'],1)
            self.assertTrue(capture['coverage']['physical_audio_complete'])
        self.assertEqual(len(emitted),2*len(captures[0]['audio_calls']))

    def test_worker_load_failure_records_final_mocked_memory_without_retry(self):
        from datetime import datetime,timedelta,timezone
        import sys
        assets={'interpreter':str(Path(sys.executable).absolute())}
        plan_path=self.folder/'plan.json';n.base.write_json(plan_path,{'synthetic':True})
        folder=self.folder/'worker'
        spec={'version':n.VERSION,'plan_path':str(plan_path),'plan_file_sha256':n._file(plan_path),
            'output_dir':str(folder),'assets':assets,'worker_seconds':30,
            'work_deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat(),'worker_sha256':n._file(n.__file__)}
        spec_path=self.folder/'spec.json';n.base.write_json(spec_path,spec)
        memory={'scope':'worker_cuda_reset_before_model_load','device_index':0,'device_name':'synthetic',
            'device_total_bytes':1000,'allocated_bytes':0,'reserved_bytes':0,'peak_allocated_bytes':10,'peak_reserved_bytes':10}
        with patch.object(n,'check_plan'),patch.object(n,'_read',side_effect=lambda p:spec if Path(p)==spec_path else {'slots':[]}), \
             patch.object(n,'cuda_memory',return_value=memory) as measure,patch.object(n,'_load_model',side_effect=RuntimeError('synthetic load failure')) as load:
            value=n.run_worker(spec_path)
        self.assertEqual(value['status'],'failed');self.assertEqual(value['model_load_attempts'],1)
        self.assertEqual(value['model_loads'],0);self.assertEqual(value['asr_calls'],0)
        self.assertEqual(value['cuda_memory_before_load'],memory);self.assertEqual(value['cuda_memory_final'],memory)
        self.assertEqual(load.call_count,1);self.assertEqual(measure.call_count,2)
        self.assertEqual(measure.call_args_list[0].kwargs,{'reset':True})
        self.assertGreater(value['seconds'],0)
        with self.assertRaisesRegex(ValueError,'already attempted'):n.run_worker(spec_path)

    def test_native_decoder_preserves_language_prefix_and_literal_marker_spelling(self):
        text='lang:ja invented <s> literal\nsecond line'
        ids=[1]+self.processor.tokenizer.encode(text,add_special_tokens=False)+[32,2]
        value=n.decode_tokens(self.processor,ids)
        self.assertEqual(value['parsed_text'],text)
        self.assertNotEqual(self.processor.batch_decode([ids],skip_special_tokens=True)[0],text)
        result=[];previous=0
        for insertion in value['special_token_insertions']:
            offset=insertion['parsed_offset'];result.extend([text[previous:offset],insertion['special_text']]);previous=offset
        result.append(text[previous:]);self.assertEqual(''.join(result),value['raw_text'])
        self.assertEqual(value['special_token_insertions'][-1]['token_ids'],[32,2])

    def test_early_eos_preserves_raw_and_restores_hooks_no_retry(self):
        model,emitted=tiny_model(self.processor,'early_eos');save=lambda v:setattr(self,'capture',v)
        with self.assertRaisesRegex(ValueError,'physical audio'):
            n.capture_call(model,self.processor,self.audio,self.request,artifact_root=self.folder,save_capture=save,deadline=time.monotonic()+30)
        self.assertEqual(len(emitted),1);self.assertEqual(self.capture['output_token_ids'][-1],2)
        self.assertIsInstance(self.capture['raw_text'],str);self.assertEqual(self.capture['generation_calls'],1)
        self.assertNotIn('_prepare_generated_length',model.__dict__);self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.audio_tower._forward_hooks)

    def test_late_eos_after_physical_domain_is_valid_native_completion(self):
        self.call('late_eos');value=self.replay(self.receipt())
        self.assertEqual(value['coverage']['stop_reason'],'eos')
        self.assertLess(value['coverage']['output_positions'],value['coverage']['total_positions'])
        self.assertTrue(value['coverage']['physical_audio_complete'])

    def test_resealed_tensor_scope_config_and_special_insertion_tampering_rejects(self):
        self.call();good=self.receipt()
        mutations=[lambda v:v['capture']['resolved_generation'].update(max_length=512),
            lambda v:v['capture']['runtime']['generation_config'].update(suppress_tokens=[32]),
            lambda v:v['capture']['audio_calls'][0].update(audio_end=4),
            lambda v:v['capture']['audio_calls'][0].update(layer_calls=1),
            lambda v:v['capture']['layout'].update(right_pad_samples=0),
            lambda v:v['capture']['special_token_insertions'][0].update(special_text='ordinary backup'),
            lambda v:v['request']['view'].update(crop_start_frame=1)]
        for mutate in mutations:
            value=deepcopy(good);mutate(value);value['receipt_sha256']=n.ec._hash({k:x for k,x in value.items() if k!='receipt_sha256'})
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.replay(value)

    def test_expired_budget_never_generates_and_restores_hooks(self):
        model,emitted=tiny_model(self.processor);save=lambda v:setattr(self,'capture',v)
        with self.assertRaises(n.WorkerDeadline):
            n.capture_call(model,self.processor,self.audio,self.request,artifact_root=self.folder,save_capture=save,deadline=time.monotonic()-1)
        self.assertEqual(emitted,[]);self.assertEqual(self.capture['generation_calls'],0)
        self.assertNotIn('_prepare_generated_length',model.__dict__);self.assertFalse(model._forward_pre_hooks)

if __name__=='__main__':unittest.main()
