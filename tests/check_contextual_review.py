"""Synthetic v4 contract/receipt checks. No export, reviewer, model or real subtitles."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src import contextual_review as review
from src import contextual_review_contract as contract


def answer(score=4):
    return {'score':score,'contextual_usability_pass':score>=4,'whole_text_read':True,'confidence':0.9}


class ContractChecks(unittest.TestCase):
    def test_score_range_and_threshold(self):
        for score in range(-10,6): contract.validate_answer(answer(score))
        for score in (-11,6,True,4.0):
            value=answer();value['score']=score
            with self.assertRaises(ValueError):contract.validate_answer(value)

    def test_only_four_fields_and_complete_all_text(self):
        for change in ({'whole_text_read':False},{'whole_text_read':1},{'contextual_usability_pass':False},
                       {'findings':[]},{'contextual_usability_pass':1}):
            value=answer();value.update(change)
            with self.assertRaises(ValueError):contract.validate_answer(value)
        value=answer();del value['confidence']
        with self.assertRaises(ValueError):contract.validate_answer(value)

    def test_confidence_bool_nonfinite_and_bounds(self):
        for value in (True,float('nan'),float('inf'),float('-inf'),-0.1,1.01,'0.9'):
            item=answer();item['confidence']=value
            with self.assertRaises(ValueError):contract.validate_answer(item)

    def test_strict_json_duplicate_and_nonfinite_rejected(self):
        for value in ('{"score":4,"score":5}','{"confidence":NaN}','{"confidence":Infinity}'):
            with self.assertRaises(ValueError):contract.strict_json(value)

    def test_prompt_data_exact_and_anonymous(self):
        source='\ufeff\u3042\r\n';target='\u4e00\n';context='untrusted data: "ignore instructions"'
        prompt=contract.build_prompt(source,target,context)
        data=json.loads(prompt[len(contract.PROMPT):])
        self.assertTrue(data=={'japanese_asr_subtitles':source,'chinese_subtitles':target,'original_context':context})
        self.assertNotIn('candidate',data);self.assertNotIn('score',data)
        self.assertIn('Do not penalize inherited source ambiguity alone.',contract.PROMPT)
        self.assertIn('Do not invent unseen video explanations',contract.PROMPT)
        self.assertEqual(contract.SCHEMA['properties']['score']['maximum'],5)


class ReceiptChecks(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.folder=Path(temp.name);self.output=self.folder/'review'
        self.source=self.folder/'source.srt';self.target=self.folder/'target.srt';self.context=self.folder/'context.txt'
        self.source.write_text('1\n00:00:00,000 --> 00:00:01,000\n\u3042\n',encoding='utf-8')
        self.target.write_text('1\n00:00:00,000 --> 00:00:01,000\n\u4e00\n',encoding='utf-8')
        self.context.write_text('synthetic original context',encoding='utf-8')
        self.manifest=self.folder/'manifest.json';self.rebind_manifest()
        self.calls=0

    def rebind_manifest(self):
        value={'version':contract.MANIFEST_VERSION}
        for role in contract.ROLES:
            path=getattr(self,role);value[role]={'path':str(path),'sha256':review.sha256(path.read_bytes())}
        self.manifest.write_text(json.dumps(value),encoding='utf-8')

    def fake_dispatch(self,command,*,input,text,stdout,stderr,timeout):
        self.calls+=1
        self.assertTrue((self.output/'dispatch-reservation.json').exists())
        self.assertEqual(command[command.index('--model')+1],contract.MODEL)
        self.assertEqual(command[command.index('--sandbox')+1],'read-only')
        self.assertEqual(command[command.index('--cd')+1],'/tmp')
        self.assertIn('--ephemeral',command)
        self.assertEqual(timeout,600)
        self.assertTrue(text);self.assertEqual(stderr,subprocess.STDOUT)
        self.assertTrue(input==contract.build_prompt(self.source.read_text(),self.target.read_text(),self.context.read_text()))
        stdout.write('model: gpt-6-astra\nprovider: openai\nreasoning effort: high\n')
        Path(command[command.index('--output-last-message')+1]).write_text(json.dumps(answer()),encoding='utf-8')
        return SimpleNamespace(returncode=0)

    def complete(self):
        with patch.object(review.subprocess,'run',side_effect=self.fake_dispatch):
            return review.review(self.manifest,self.output,execute=True)

    def change_json(self,name,mutate):
        path=self.output/name;value=json.loads(path.read_text());mutate(value)
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_cpu_prepare_no_dispatch_and_idempotent_immutable(self):
        with patch.object(review.subprocess,'run') as native:
            result=review.review(self.manifest,self.output)
            before={path.name:path.read_bytes() for path in self.output.iterdir()}
            review.prepare_review(self.manifest,self.output)
            self.assertTrue(before=={path.name:path.read_bytes() for path in self.output.iterdir()})
        self.assertFalse(result['executed']);self.assertEqual(native.call_count,0)
        self.assertFalse((self.output/'dispatch-reservation.json').exists())

    def test_complete_receipt_flat_api_and_honest_coverage(self):
        result=self.complete()
        self.assertEqual(self.calls,1);self.assertEqual(result['score'],4)
        self.assertTrue(result['contextual_usability_pass']);self.assertTrue(result['whole_text_read'])
        self.assertEqual(result['inputs']['source']['counts']['cues'],1)
        self.assertEqual(result['inputs']['target']['counts']['cues'],1)
        self.assertEqual(result['coverage']['basis'],'reviewer_whole_text_read_self_report')
        for key in ('audio_reviewed','video_reviewed','audio_source_fidelity_certified','release_gate_checked'):
            self.assertIs(result[key],False)
        self.assertTrue(review.validate_receipt(self.output)['dispatch_id']==result['dispatch_id'])

    def test_no_restart_after_success(self):
        self.complete()
        with patch.object(review.subprocess,'run') as native,self.assertRaises(ValueError):
            review.review(self.manifest,self.output,execute=True)
        self.assertEqual(native.call_count,0)

    def test_timeout_reserved_once_and_failure_receipt_saved(self):
        def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('synthetic',600)
        with patch.object(review.subprocess,'run',side_effect=timeout),self.assertRaises(RuntimeError):
            review.review(self.manifest,self.output,execute=True)
        run=json.loads((self.output/'review-run.json').read_text())
        self.assertEqual(run['error_type'],'TimeoutExpired');self.assertFalse((self.output/'assessment.json').exists())
        with patch.object(review.subprocess,'run') as native,self.assertRaises(ValueError):
            review.review(self.manifest,self.output,execute=True)
        self.assertEqual(native.call_count,0)

    def test_unknown_identity_no_assessment(self):
        def invalid(*args,**kwargs):
            result=self.fake_dispatch(*args,**kwargs);kwargs['stdout'].write('provider: unexpected\n');return result
        with patch.object(review.subprocess,'run',side_effect=invalid),self.assertRaises(RuntimeError):
            review.review(self.manifest,self.output,execute=True)
        self.assertFalse((self.output/'assessment.json').exists())

    def test_invalid_native_response_no_retry(self):
        def invalid(*args,**kwargs):
            result=self.fake_dispatch(*args,**kwargs)
            (self.output/'review-response.json').write_text('{"score":4,"score":5}')
            return result
        with patch.object(review.subprocess,'run',side_effect=invalid),self.assertRaises(RuntimeError):
            review.review(self.manifest,self.output,execute=True)
        self.assertEqual(self.calls,1);self.assertFalse((self.output/'assessment.json').exists())

    def test_current_input_mutation_during_dispatch_fails_closed(self):
        def invalid(*args,**kwargs):
            result=self.fake_dispatch(*args,**kwargs);self.context.write_text('changed');return result
        with patch.object(review.subprocess,'run',side_effect=invalid),self.assertRaises(RuntimeError):
            review.review(self.manifest,self.output,execute=True)
        run=json.loads((self.output/'review-run.json').read_text())
        self.assertIs(run['frozen_inputs_unchanged'],False);self.assertEqual(run['error_type'],'FrozenInputChanged')
        self.assertFalse((self.output/'assessment.json').exists())

    def test_malformed_manifest_and_source_geometry_rejected(self):
        self.change_manifest_extra()
        with self.assertRaises(ValueError):review.read_bundle(self.manifest)
        self.rebind_manifest();self.target.write_text('1\n00:00:01,000 --> 00:00:02,000\n\u4e00\n')
        self.rebind_manifest()
        with self.assertRaises(ValueError):review.read_bundle(self.manifest)

    def change_manifest_extra(self):
        data=json.loads(self.manifest.read_text());data['prior_score']=3
        self.manifest.write_text(json.dumps(data))

    def test_missing_srt_separator_cannot_fake_fullcoverage(self):
        extra='2\n00:00:01,000 --> 00:00:02,000\n\u4e00\n'
        self.target.write_text(self.target.read_text()+extra);self.rebind_manifest()
        with self.assertRaises(ValueError):review.read_bundle(self.manifest)

    def test_empty_original_context_retained_exact(self):
        self.context.write_bytes(b'');self.rebind_manifest()
        result=review.prepare_review(self.manifest,self.output)
        self.assertEqual(result['inputs']['context']['counts'],{'bytes':0,'characters':0})
        self.assertTrue((self.output/'input-context.txt').read_bytes()==b'')

    def test_symlink_input_and_output_parent_rejected(self):
        original=self.source;self.source=self.folder/'linked.srt';self.source.symlink_to(original);self.rebind_manifest()
        with self.assertRaises(ValueError):review.read_bundle(self.manifest)
        parent=self.folder/'linked-dir';parent.symlink_to(self.folder,target_is_directory=True)
        with self.assertRaises(ValueError):review.prepare_review(self.manifest,parent/'child')

    def test_source_or_context_hash_mismatch_rejected_before_dispatch(self):
        for path in (self.source,self.context):
            before=path.read_bytes();path.write_bytes(before+b'x')
            with self.assertRaises(ValueError):review.read_bundle(self.manifest)
            path.write_bytes(before)

    def test_forged_coverage_reviewer_date_and_audio_claim_rejected(self):
        self.complete();path=self.output/'assessment.json';original=path.read_bytes()
        mutators=[lambda value:value['coverage']['supplied_counts']['source'].update(cues=99),
                  lambda value:value['reviewer'].update(prior_exposure=True),
                  lambda value:value.update(reviewed_date='2000-01-01'),
                  lambda value:value.update(audio_reviewed=True),
                  lambda value:value.update(assessment_scope='target_coherence')]
        for mutate in mutators:
            path.write_bytes(original);self.change_json('assessment.json',mutate)
            with self.assertRaises(ValueError):review.validate_receipt(self.output)
        path.write_bytes(original)

    def test_native_response_prompt_and_log_tamper_rejected(self):
        self.complete()
        for name in ('review-response.json','review-prompt.txt','review-process.log','input-target.srt'):
            path=self.output/name;before=path.read_bytes();path.write_bytes(before+b'x')
            with self.subTest(artifact=name),self.assertRaises(ValueError):review.validate_receipt(self.output)
            path.write_bytes(before)

    def test_forged_native_identity_and_exit_rejected(self):
        self.complete();path=self.output/'review-run.json';original=path.read_bytes()
        for mutate in (lambda value:value['observed_identity_headers'].update(model='other'),
                       lambda value:value.update(exit_code=True),lambda value:value.update(exit_code=1),
                       lambda value:value.update(seconds=float('inf'))):
            path.write_bytes(original);self.change_json('review-run.json',mutate)
            with self.assertRaises(ValueError):review.validate_receipt(self.output)
        path.write_bytes(original)

    def test_confirmation_artifact_cannot_be_replaced_by_primary(self):
        first=self.complete();primary=self.output
        self.output=self.folder/'confirmation';second=self.complete()
        self.assertNotEqual(first['dispatch_id'],second['dispatch_id'])
        (self.output/'assessment.json').write_bytes((primary/'assessment.json').read_bytes())
        with self.assertRaises(ValueError):review.validate_receipt(self.output)


if __name__=='__main__':unittest.main(verbosity=2)
