"""Synthetic contextual gate checks; no real subtitle or external review inputs."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src import contextual_release as gate
from src.release import deliverables
import main


class ContextualGateChecks(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.folder=Path(temp.name);self.unit=self.folder/'unit';final=self.unit/'final';final.mkdir(parents=True)
        (final/'clip.wav').write_bytes(b'synthetic media metadata only')
        for name,text in [('clip.srt','\u3042'),('clip.zh.srt','\u4e00')]:
            (final/name).write_text('1\n00:00:00,000 --> 00:00:01,000\n'+text+'\n',encoding='utf-8')
        self.path=self.unit/'review'/'assessment.json'
        self.assessment=gate.prepare_assessment(self.unit,self.path,writer_model='gemma4-31b-qat-q4')
        files=self.assessment['files']
        inputs={'source':{'path':str(final/'clip.srt'),'sha256':files['final/clip.srt'],'counts':{'cues':1}},
                'target':{'path':str(final/'clip.zh.srt'),'sha256':files['final/clip.zh.srt'],'counts':{'cues':1}},
                'context':{'path':str(self.folder/'context.txt'),'sha256':'c'*64,'counts':{'characters':5}}}
        self.primary={'score':4,'contextual_usability_pass':True,'inputs':deepcopy(inputs),
            'dispatch_id':'00000000-0000-0000-0000-000000000001',
            'started_utc':'2026-09-14T01:00:00+00:00','finished_utc':'2026-09-14T01:01:00+00:00',
            'receipt_hashes':{'preparation.json':'a'*64,'assessment.json':'b'*64}}
        self.confirmation={**deepcopy(self.primary), 'dispatch_id':'00000000-0000-0000-0000-000000000002',
            'started_utc':'2026-09-14T01:02:00+00:00','finished_utc':'2026-09-14T01:03:00+00:00',
            'receipt_hashes':{'preparation.json':'d'*64,'assessment.json':'e'*64}}
        self.assessment.update(score=4,contextual_reviews=[{'target':'final/clip.zh.srt',
            'manifest':str(self.folder/'manifest.json'),'primary':'primary','confirmation':'confirmation',
            'primary_receipt_hashes':deepcopy(self.primary['receipt_hashes']),
            'confirmation_receipt_hashes':deepcopy(self.confirmation['receipt_hashes'])}])

    def lookup(self,directory,manifest):
        self.assertEqual(manifest,str(self.folder/'manifest.json'))
        return deepcopy({'primary':self.primary,'confirmation':self.confirmation}[directory])

    def validate(self,side_effect=None):
        with patch.object(gate,'validate_receipt',side_effect=side_effect or self.lookup):
            return gate.validate_assessment(self.unit,self.assessment)

    def test_primary_plus_confirmation_and_frozen_receipts_pass(self):
        self.assertEqual(self.validate(),[])

    def test_prepared_template_is_never_approved(self):
        raw=json.loads(self.path.read_text())
        with patch.object(gate,'validate_receipt') as reviewer:
            self.assertTrue(gate.validate_assessment(self.unit,raw))
        self.assertEqual(reviewer.call_count,0)

    def test_score_minimum_of_both_and_no_six_or_bool(self):
        self.primary['score']=5
        self.assertEqual(self.validate(),[])
        for score in (5,6,True):
            self.assessment['score']=score;self.assertTrue(self.validate())

    def test_failed_or_incomplete_confirmation_blocks(self):
        for change in ({'score':3,'contextual_usability_pass':False},{'contextual_usability_pass':False}):
            self.confirmation.update(change);self.assertTrue(self.validate())
        def incomplete(*_):raise ValueError('Synthetic incomplete receipt')
        self.assertTrue(self.validate(incomplete))

    def test_confirmation_different_context_source_or_target_blocks(self):
        for role in ('source','target','context'):
            before=self.confirmation['inputs'][role]['sha256']
            self.confirmation['inputs'][role]['sha256']='f'*64
            self.assertTrue(self.validate());self.confirmation['inputs'][role]['sha256']=before

    def test_both_reviews_wrong_final_hash_block(self):
        for receipt in (self.primary,self.confirmation):receipt['inputs']['target']['sha256']='f'*64
        self.assertTrue(self.validate())

    def test_reused_dispatch_and_overlapping_confirmation_block(self):
        original=self.confirmation['dispatch_id'];self.confirmation['dispatch_id']=self.primary['dispatch_id']
        self.assertTrue(self.validate());self.confirmation['dispatch_id']=original
        self.confirmation['started_utc']='2026-09-14T01:00:30+00:00'
        self.assertTrue(self.validate())

    def test_mutated_internally_consistent_receipt_rejected_by_assessment_pins(self):
        self.primary['receipt_hashes']['assessment.json']='f'*64
        self.assertTrue(self.validate())
        self.assessment['contextual_reviews'][0].pop('primary_receipt_hashes')
        self.assertTrue(self.validate())

    def test_missing_or_unreviewed_second_target_blocks(self):
        self.assessment['contextual_reviews']=[];self.assertTrue(self.validate())
        self.set_up_evidence_again()
        (self.unit/'final'/'clip.en.srt').write_text('1\n00:00:00,000 --> 00:00:01,000\nA\n')
        self.assessment['files']=deliverables(self.unit)
        self.assertTrue(self.validate())

    def set_up_evidence_again(self):
        self.assessment['contextual_reviews']=[{'target':'final/clip.zh.srt',
            'manifest':str(self.folder/'manifest.json'),'primary':'primary','confirmation':'confirmation',
            'primary_receipt_hashes':deepcopy(self.primary['receipt_hashes']),
            'confirmation_receipt_hashes':deepcopy(self.confirmation['receipt_hashes'])}]

    def test_unknown_target_and_duplicate_target_mapping_block(self):
        self.assessment['contextual_reviews'][0]['target']='final/other.zh.srt'
        self.assertTrue(self.validate());self.set_up_evidence_again()
        self.assessment['contextual_reviews'].append(deepcopy(self.assessment['contextual_reviews'][0]))
        self.assertTrue(self.validate())

    def test_source_target_geometry_mismatch_blocks_before_reviews(self):
        path=self.unit/'final'/'clip.srt';path.write_text(path.read_text().replace('00:00:01,000','00:00:02,000'))
        self.assessment['files']=deliverables(self.unit)
        with patch.object(gate,'validate_receipt') as reviewer:
            self.assertTrue(gate.validate_assessment(self.unit,self.assessment))
        self.assertEqual(reviewer.call_count,0)

    def test_current_files_mutated_during_receipt_validation_block(self):
        def mutate(directory,manifest):
            result=self.lookup(directory,manifest)
            if directory=='confirmation':
                path=self.unit/'final'/'clip.zh.srt';path.write_bytes(path.read_bytes()+b'\n')
            return result
        self.assertTrue(self.validate(mutate))

    def test_timing_warning_visible_but_not_material_translation_failure(self):
        for filename in ('clip.srt','clip.zh.srt'):
            path=self.unit/'final'/filename;path.write_text(path.read_text().replace('00:00:01,000','00:00:00,200'))
        self.assessment['files']=deliverables(self.unit)
        for receipt in (self.primary,self.confirmation):
            for role,filename in (('source','clip.srt'),('target','clip.zh.srt')):
                receipt['inputs'][role]['sha256']=self.assessment['files']['final/'+filename]
        self.assertEqual(self.validate(),[])

    def test_wrong_scope_writer_and_audio_certification_block(self):
        for field,value in [('benchmark','subtitle-quality-v3'),('assessment_scope','source_verified'),
                            ('writer_model','gpt-6-astra'),('writer_execution','external'),
                            ('source_accuracy_verified',True),('playback_verified',True)]:
            before=self.assessment[field];self.assessment[field]=value
            self.assertTrue(self.validate());self.assessment[field]=before

    def test_symlinked_final_file_blocks(self):
        path=self.unit/'final'/'clip.zh.srt';saved=self.folder/'saved.srt';path.rename(saved);path.symlink_to(saved)
        self.assertTrue(self.validate())

    def test_full_synthetic_receipts_validate_through_gate_without_adapter_mock(self):
        from src import contextual_review as native
        context=self.folder/'context.txt';context.write_text('synthetic context')
        manifest=self.folder/'manifest.json'
        spec={'version':'contextual-review-bundle-v1'}
        for role,path in (('source',self.unit/'final'/'clip.srt'),
                          ('target',self.unit/'final'/'clip.zh.srt'),('context',context)):
            spec[role]={'path':str(path),'sha256':native.sha256(path.read_bytes())}
        manifest.write_text(json.dumps(spec))
        def fake(command,**kwargs):
            kwargs['stdout'].write('model: gpt-6-astra\nprovider: openai\nreasoning effort: high\n')
            path=Path(command[command.index('--output-last-message')+1])
            path.write_text(json.dumps({'score':4,'contextual_usability_pass':True,
                                        'whole_text_read':True,'confidence':0.9}))
            return SimpleNamespace(returncode=0)
        with patch.object(native.subprocess,'run',side_effect=fake) as dispatched:
            primary=native.review(manifest,self.folder/'native-primary',execute=True)
            confirmation=native.review(manifest,self.folder/'native-confirmation',execute=True)
        self.assertEqual(dispatched.call_count,2)
        evidence=self.assessment['contextual_reviews'][0]
        evidence.update(primary=primary['output_dir'],confirmation=confirmation['output_dir'],
                        primary_receipt_hashes=primary['receipt_hashes'],
                        confirmation_receipt_hashes=confirmation['receipt_hashes'])
        self.assertEqual(gate.validate_assessment(self.unit,self.assessment),[])

    def test_cli_contextual_scope_cannot_package(self):
        args=SimpleNamespace(unit_dir=self.unit,assessment=self.path,writer_model=None,scope='contextual_subtitles',
                             prepare=False,check=False,destination=self.folder/'released')
        with patch('src.release.release_unit') as package:
            self.assertEqual(main.cmd_release(args),1)
        self.assertEqual(package.call_count,0)

    def test_cli_contextual_check_routes_only_to_contextual_gate(self):
        self.path.write_text(json.dumps(self.assessment))
        args=SimpleNamespace(unit_dir=self.unit,assessment=self.path,writer_model=None,scope='contextual_subtitles',
                             prepare=False,check=True,destination=self.folder/'released')
        with patch.object(gate,'validate_assessment',return_value=[]) as contextual, \
             patch('src.release.validate_coherence_assessment') as legacy:
            self.assertEqual(main.cmd_release(args),0)
        self.assertEqual(contextual.call_count,1);self.assertEqual(legacy.call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
