"""Synthetic masked-source screen, actual shared native capture, no local models."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import masked_source_screen as screen
from src import auto_source_draft_requests as auto
from src import compact_raw_draft_requests as compact
from src import evidence_context as ec
from src import local_gemma_native as native
from src.workflow_state import file_hash, fingerprint, write_json
from tests.test_masked_source_draft_requests import MaskedSourceDraftFixture
from tests.test_local_gemma_native import FakeResponse


class MaskedSourceScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name); self.folder = self.directory/'masked-draft'; self.parent = self.directory/'acquisition'
        self.parent.mkdir(); self.b_parent = self.directory/'B'; self.b_parent.mkdir()
        f = MaskedSourceDraftFixture(); acoustic = f.auto.f
        for row in acoustic.source: row['text'] = row['text'].strip()
        texts = {row['index']:row['text'] for row in acoustic.source}
        for row in acoustic.observations:
            if row.get('origin') == 'immutable_first_asr':
                row['text'] = texts[row['owner_id']]
                for reading in acoustic.frames[str(row['owner_id'])]['readings']:
                    if reading['observation_id'] == row['observation_id']: reading['quote'] = row['text']
        acoustic.rebuild(); self.source = acoustic.source; self.context = acoustic.context.encode()
        b = compact.build_writer_request(acoustic.pack,acoustic.bound,acoustic.identities,projection=acoustic.projection)
        self.b = b; self.observations = deepcopy(f.observations)
        self.observations['source_rows_sha256'] = ec._hash(self.source)
        self.make_acquisition()
        self.request = screen.contract.build_writer_request(self.b,self.observations)
        for name,value in [('request.json',self.b),('source.json',self.source)]: write_json(self.b_parent/name,value)
        (self.b_parent/'original-context.txt').write_bytes(self.context)
        self.plan = self.directory/'plan.md'; self.plan.write_text('Synthetic local-only plan')
        self.profile = self.directory/'profile.json'; write_json(self.profile,{'synthetic':'profile'})
        self.binary = self.directory/'server'; self.binary.write_bytes(b'synthetic server')
        self.weights = self.directory/'weights.gguf'; self.weights.write_bytes(b'synthetic weights')
        self.deployment = {'model':screen.SETTINGS.model,'sha256':screen.SETTINGS.model_sha256,
            'weights':str(self.weights),'server_binary':str(self.binary),'context_size':163840,'full_swa':False,
            'gpu_layers':99,'cpu_moe_layers':0,'threads':4,'admin_url':'http://127.0.0.1:8089/admin','restore_model':'original'}
        self.identity = {'model_alias':screen.SETTINGS.model,'model_sha256':screen.SETTINGS.model_sha256,
            'model_path':str(self.weights),'context_size':163840,'full_swa':False,'pid':1234,
            'server_binary_sha256':file_hash(self.binary),'chat_template_sha256':hashlib.sha256(b'synthetic template').hexdigest()}
        self.output = {'owners':{str(i):{'chinese':f' Synthetic exact {i} e\u0301\nsecond line'} for i in range(1,67)}}
        self.posts = self.closed = 0; self.backend = 'original'; self.reasoning = ''
        self.fail_transport = self.exhaust_cleanup = False
        changes = [patch.object(screen,'ACQUISITION_PARENT',self.parent),patch.object(screen,'B_PARENT',self.b_parent),
            patch.object(screen,'B_REQUEST_SHA256',fingerprint(self.b)),
            patch.object(screen,'SOURCE',screen.rows_hash(self.source)),patch.object(screen,'CONTEXT',hashlib.sha256(self.context).hexdigest()),
            patch.object(screen,'PLAN',self.plan),patch.object(screen,'PROFILE',self.profile),
            patch.object(screen,'recipe',side_effect=lambda:deepcopy(self.deployment)),
            patch.object(screen,'backend_state',side_effect=lambda _:self.backend),patch.object(screen,'pid_gone',return_value=True),
            patch.object(screen.lb,'temporary_local_writer',side_effect=self.owned),
            patch.object(native,'_request_json',side_effect=self.rpc),patch.object(native.requests,'post',side_effect=self.post)]
        for change in changes: change.start(); self.addCleanup(change.stop)

    def make_acquisition(self):
        """Invent a self-contained, already replayed native completion receipt."""
        attention=self.observations['audio_attention']
        attention['source_identity']['files']={str((screen.ROOT/'src/qwen_audio_attention.py').resolve()):file_hash(screen.ROOT/'src/qwen_audio_attention.py')}
        config=self.directory/'asr-model/config.json';write_json(config,{'synthetic_model':True})
        attention['model_config']={'path':str(config),'sha256':file_hash(config)}
        for number,control in enumerate(attention['control_receipts'],1):
            path=self.parent/'worker/receipts'/f'{number:02d}-{control["slot_id"]}.json'
            write_json(path,{'synthetic_control':control['slot_id']});control.update(path=str(path),sha256=file_hash(path))
        geometry=[]
        for number,row in enumerate(self.observations['observations'],3):
            path=self.parent/'worker/receipts'/f'{number:02d}-owner-{row["owner_id"]:03d}.json'
            write_json(path,{'synthetic_owner':row['owner_id']});row['receipt'].update(path=str(path),sha256=file_hash(path))
            geometry.append({k:deepcopy(row[k]) for k in ('owner_id','owner_ts_line','owner_start_frame','owner_end_frame',
                'crop_start_frame','crop_end_frame','sample_rate','clamped_tail_frames','vad')})
        self.observations['observations_sha256']=ec._hash(self.observations['observations'])
        plan={'version':'masked-source-plan-1','parent_plan':{k:deepcopy(self.observations[k]) for k in
              ('source_rows_sha256','original_mono_sha256','original_mono_frames','sample_rate')},
            'execution_identity':deepcopy(self.observations['execution_identity']),'attention_policy':attention['policy'],
            'attention_source':attention['source_identity'],'encoder_config':attention['encoder_config'],'model_config':attention['model_config']}
        plan['parent_plan']['geometry']=geometry;plan['plan_sha256']=fingerprint(plan);self.observations['plan_sha256']=plan['plan_sha256']
        write_json(self.parent/'native-plan.json',plan);write_json(self.parent/'observations.json',self.observations)
        write_json(self.parent/'assets.json',{'synthetic':True});write_json(self.parent/'worker/worker-result.json',{'synthetic_native_complete':True})
        write_json(self.parent/'lifecycle.json',{'restored':True,'owned_worker_stopped':True,'error_type':None})
        write_json(self.parent/'input-proof.json',{'source_bindings':{'source_provenance':{'synthetic':True},
            'detector_provenance':{'mono_sha256':self.observations['original_mono_sha256']}}})
        pins={str((screen.ROOT/p).resolve()):file_hash(screen.ROOT/p) for p in
              ('scripts/masked_source_acquisition.py','src/masked_source_native.py','src/qwen_audio_attention.py')}
        pins.update(attention['source_identity']['files']);pins[str(config)]=file_hash(config)
        reg={'version':'masked-source-acquisition-1','pins':pins};reg['registration_sha256']=fingerprint(reg)
        write_json(self.parent/'registration.json',reg)
        state={'version':'masked-source-acquisition-1','status':'complete','passed':True,'original_backend_restored':True,
            'local_seconds':12.,'work_seconds':10.,'registration_sha256':reg['registration_sha256']}
        write_json(self.parent/'diagnostic.json',state)
        controls={name:{'passed':True,'parsed_nonempty':False,'protocol_category':'empty_raw'} for name in ('silence','noise')}
        result={'version':'masked-source-acquisition-1','passed':True,'reason_code':'complete_masked_observations',
            'asr_calls':68,'control_calls':2,'real_audio_calls':66,'model_loads':1,'new_recap_calls':0,'candidate_generated':False,
            'score':None,'accuracy_verified':False,'independent_recognizer_vote':False,'original_backend_restored':True,
            'native_complete':True,'controls':controls,'local_seconds':12.,'work_seconds':10.,
            'runtime_backend':{'asr_backend':'transformers','audio_backend':'sdpa','model_dir':str(config.parent),
                'model_config':attention['model_config'],'audio_encoder_config':attention['encoder_config'],
                'thinker_text_backend':'sdpa','device':'cuda:0','dtype':'torch.float16'},
            'attention_policy':attention['policy'],'attention_source_sha256':fingerprint(attention['source_identity']),
            'source_rows_sha256':self.observations['source_rows_sha256'],'geometry_sha256':fingerprint(geometry),
            'observation_envelope_sha256':fingerprint(self.observations)}
        result['artifacts']={str(p.resolve()):file_hash(p) for p in self.parent.rglob('*') if p.is_file()}
        write_json(self.parent/'result.json',result)

    def rpc(self,url,payload=None,**kwargs):
        if url.endswith('/props'):
            return {'model_alias':screen.SETTINGS.model,'model_path':str(self.weights),'chat_template':'synthetic template',
                    'default_generation_settings':{'n_ctx':163840}}
        if url.endswith('/apply-template'):
            self.assertEqual(payload['reasoning_budget_tokens'],0); self.assertEqual(payload['reasoning_effort'],'none')
            self.assertEqual(payload['max_tokens'],16384); self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(json.loads(payload['messages'][1]['content']),self.request['body'])
            self.assertNotIn('audit',json.loads(payload['messages'][1]['content']))
            return {'prompt':'synthetic full masked-source prompt'}
        if url.endswith('/tokenize'): return {'tokens':list(range(100))}
        self.fail('Unexpected native RPC')

    def post(self,url,**kwargs):
        self.posts += 1; self.assertEqual(url,native.ENDPOINT)
        if self.fail_transport: raise RuntimeError('Synthetic transport failure')
        if self.exhaust_cleanup:
            s = screen.state(self.folder); screen.save(self.folder,cleanup_seconds=420,local_seconds=s['work_seconds']+420)
        return FakeResponse(json.dumps(self.output,ensure_ascii=False),reasoning=self.reasoning)

    @contextmanager
    def owned(self,weights,binary,**kwargs):
        self.assertEqual((weights,binary),(str(self.weights),str(self.binary)))
        self.assertEqual(kwargs['context_size'],163840); self.assertTrue(kwargs['restore_previous'])
        self.backend = 'owned'
        try: yield native.ENDPOINT,deepcopy(self.identity)
        finally: self.closed += 1; self.backend = 'original'

    def local(self): screen.prepare(self.folder); return screen.execute_local(self.folder)

    def fake_reviews(self):
        self.review_calls=[]; self.next_score=4; self.review_fails=False; self.receipts={}; self.prior=None
        write_json(self.b_parent/'registration.json',{'baseline_review':'/synthetic/baseline','evidence_paths':['/synthetic/original736.json']})
        baseline={'output_dir':'/synthetic/baseline','purpose':'baseline','score':2}
        def validate(value,arm):
            self.assertEqual(value['pool_version'],arm['pool_version'])
            for key in ('source','raw','context','target'):
                self.assertEqual(value['inputs'][key]['sha256'],arm[('source' if key=='raw' else key)+'_sha256'])
        def bundle(source,target,raw,paths,context,directory):
            self.assertEqual(source,raw); self.assertEqual(source,self.source)
            self.assertEqual(paths,[Path('/synthetic/original736.json')])
            value={'pool_version':screen.POOL,'rows':{'source':source,'target':target,'raw':raw},
                'inputs':{k:{'sha256':v} for k,v in [('source',screen.SOURCE),('raw',screen.SOURCE),
                    ('context',screen.CONTEXT),('target',screen.rows_hash(target))]}}
            write_json(directory/'manifest.json',value)
        def review_once(manifest,directory,**kwargs):
            self.review_calls.append((manifest,directory,kwargs)); write_json(directory/'dispatch-reservation.json',{'reserved':True})
            if self.review_fails: raise RuntimeError('Synthetic external failure')
            value=screen.read(Path(manifest)); receipt={'pool_version':screen.POOL,'inputs':value['inputs'],
                'purpose':kwargs['purpose'],'score':self.next_score,'output_dir':str(directory),'receipt_hashes':{},
                'dispatch_id':'dispatch-'+str(len(self.review_calls)),'started_utc':'2026-09-16T00:00:00+00:00'}
            if kwargs['purpose']=='confirmation': receipt['dependency']={'dispatch_id':self.receipts[str(kwargs['primary_review'])]['dispatch_id']}
            self.receipts[str(directory)]=receipt; return deepcopy(receipt)
        catalog=SimpleNamespace(validate_score=validate,load_score_index=Mock(return_value={'reviews':[],'records':[],'load_errors':[]}),
            duplicates=Mock(side_effect=lambda *_:deepcopy(self.prior)))
        helpers=SimpleNamespace(EXTRA_SCORES=[],_baseline=Mock(return_value=baseline),
            er=SimpleNamespace(prepare_bundle=bundle,read_bundle=screen.read,validate_receipt=lambda p:deepcopy(self.receipts[str(p)])),
            sr=SimpleNamespace(review_once=review_once))
        confirmations=SimpleNamespace(_prior_confirmation=Mock(return_value=None))
        change=patch.object(screen,'scoring',return_value=(catalog,helpers,confirmations));change.start();self.addCleanup(change.stop)
        return catalog,helpers,confirmations

    def test_local_exact865_new_version_au_layout_and_no_scoring_dependency(self):
        with patch.object(screen,'scoring',side_effect=AssertionError('Local generation cannot consult scoring')):
            result=self.local(); reg=screen.validate_registration(self.folder)
        self.assertEqual(result['status'],'local_complete');self.assertEqual((self.posts,self.closed),(1,1))
        self.assertTrue(result['original_backend_restored']);self.assertFalse(result['qualified'])
        self.assertEqual(reg['coverage']['total_observation_records'],865)
        request=screen.read(self.folder/'request.json')
        self.assertEqual(request['body']['compact_evidence'],self.b['body'])
        self.assertEqual(request['body']['automatic_source'],self.observations)
        self.assertEqual(set(request['body']),{'compact_evidence','automatic_source'})
        self.assertFalse((self.folder/'evaluation').exists());self.assertFalse((self.b_parent/'registration.json').exists())
        self.assertFalse(any('score.json' in p for p in reg['pins']))
        self.assertEqual(screen.read(self.folder/'draft.json')[0]['text'],self.output['owners']['1']['chinese'])
        self.assertEqual(result['acquisition_local_seconds'],12.)
        self.assertAlmostEqual(result['acquisition_plus_candidate_local_seconds'],12.+result['local_seconds'])
        with self.assertRaises(ValueError):screen.execute_local(self.folder)
        self.assertEqual(self.posts,1)

    def test_failed_control_or_native_completion_blocks_before_writer(self):
        original=screen.read(self.parent/'result.json')
        for field in ('control','native_complete'):
            value=deepcopy(original)
            if field=='control':value['controls']['noise']['passed']=False
            else:value['native_complete']=False
            write_json(self.parent/'result.json',value)
            with self.subTest(field=field),self.assertRaises(ValueError):screen.prepare(self.directory/field)
            self.assertEqual((self.posts,self.closed),(0,0))
        write_json(self.parent/'result.json',original)

    def test_resealed_copied_source_cannot_replace_pinned_native_origin(self):
        screen.prepare(self.folder);observations=deepcopy(self.observations)
        next(row for row in observations['observations'] if row['native_status']=='complete')['text']+=' changed'
        observations['observations_sha256']=ec._hash(observations['observations'])
        request=screen.contract.build_writer_request(self.b,observations)
        proof=screen.read(self.folder/'acquisition-proof.json');proof['envelope_sha256']=fingerprint(observations)
        for name,value in [('observations.json',observations),('request.json',request),
                           ('acquisition-proof.json',proof),('native-request.json',screen.prepared(request))]:write_json(self.folder/name,value)
        reg=screen.read(self.folder/'registration.json')
        for name in screen.COPIES:reg['pins'][str((self.folder/name).resolve())]=file_hash(self.folder/name)
        reg['registration_sha256']=fingerprint({k:v for k,v in reg.items() if k!='registration_sha256'})
        write_json(self.folder/'registration.json',reg);screen.save(self.folder,registration_sha256=reg['registration_sha256'])
        with self.assertRaisesRegex(ValueError,'native origin'):screen.execute_local(self.folder)
        self.assertEqual((self.posts,self.closed),(0,0))

    def test_failed_transport_restores_and_cannot_regenerate(self):
        self.fail_transport=True;screen.prepare(self.folder)
        with self.assertRaises(RuntimeError):screen.execute_local(self.folder)
        self.assertEqual((self.posts,self.closed),(1,1));self.assertTrue(screen.state(self.folder)['original_backend_restored'])
        with self.assertRaises(ValueError):screen.execute_local(self.folder)
        self.assertEqual(self.posts,1);self.assertFalse((self.folder/'completion.json').exists())

    def test_cleanup_overrun_still_closes_owned_backend_and_fails(self):
        self.exhaust_cleanup=True;screen.prepare(self.folder)
        with self.assertRaises(TimeoutError):screen.execute_local(self.folder)
        self.assertEqual((self.posts,self.closed),(1,1));self.assertEqual(self.backend,'original')
        self.assertTrue(screen.state(self.folder)['original_backend_restored']);self.assertEqual(screen.state(self.folder)['status'],'failed')

    def test_optional_final_review_and_confirmation_export_only_original_pool(self):
        self.local();self.fake_reviews();primary=screen.review(self.folder);confirmed=screen.review(self.folder,confirmation=True)
        self.assertEqual(primary['score'],4);self.assertTrue(confirmed['benchmark_confirmed']);self.assertFalse(confirmed['qualified'])
        self.assertEqual([row[2]['purpose'] for row in self.review_calls],['candidate','confirmation']);self.assertEqual(self.posts,1)
        self.assertEqual(screen.state(self.folder)['status'],'local_complete')
        with self.assertRaises(ValueError):screen.review(self.folder,confirmation=True)

    def test_external_failure_preserves_local_output_and_no_score_retry(self):
        self.local();before=file_hash(self.folder/'draft.utterances.srt');self.fake_reviews();self.review_fails=True
        with self.assertRaises(RuntimeError):screen.review(self.folder)
        self.assertEqual(screen.state(self.folder)['status'],'local_complete');self.assertEqual(file_hash(self.folder/'draft.utterances.srt'),before)
        with self.assertRaises(ValueError):screen.review(self.folder)
        self.assertEqual(len(self.review_calls),1);self.assertEqual(self.posts,1)

    def test_identical_completed_primary_reused_without_external_dispatch(self):
        self.local();self.fake_reviews();draft=screen.read(self.folder/'draft.json')
        self.prior={'pool_version':screen.POOL,'inputs':{k:{'sha256':v} for k,v in [('source',screen.SOURCE),('raw',screen.SOURCE),
            ('context',screen.CONTEXT),('target',screen.rows_hash(draft))]},'output_dir':'/synthetic/existing',
            'purpose':'candidate','score':3,'dispatch_id':'earliest','receipt_hashes':{}}
        self.receipts[self.prior['output_dir']]=deepcopy(self.prior)
        score=screen.review(self.folder);self.assertTrue(score['reused_completed_review']);self.assertEqual(self.review_calls,[])


if __name__=='__main__':unittest.main()
