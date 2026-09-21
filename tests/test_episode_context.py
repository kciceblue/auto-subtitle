"""Offline full-source preservation, stage isolation and strict output contracts."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from main import build_parser, cmd_pipeline
from src import quality, translate, workflow
from src.config import TranslateConfig
from src.entities import EntityPlan, EntityReplacement, EntityRowPlan
from src.episode_context import MAX_SOURCE_JSON_BYTES, episode_source_prefix
from src.evidence import CueEvidence
from src.translate import SrtBlock, StreamResult
from src.workflow_state import fingerprint


class EpisodeContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root/'requests.json'
        self.source = [SrtBlock(i, f'00:00:0{i},000 --> 00:00:0{i+1},000', text)
                       for i,text in enumerate(('ユキ。','元気です。','明日です。','午後です。',
                                                 '会いました。','待っています。','遠くの町です。'),1)]
        self.evidence = [CueEvidence(b.index,b.text,'A') for b in self.source]
        self.config = TranslateConfig(translation_episode_context=True,stage='translation',max_tokens=2048)

    def context_document(self, instruction):
        section = instruction.split('READ-ONLY EPISODE SOURCE:\n',1)[1]
        return json.loads(section.split('\nEND READ-ONLY CONTEXT. TARGET TASK:',1)[0])

    def test_default_off_cli_guard_and_collect(self):
        self.assertFalse(TranslateConfig().translation_episode_context)
        self.assertFalse(build_parser().parse_args(['pipeline']).translation_episode_context)
        args = build_parser().parse_args(['pipeline','--translation-episode-context'])
        with patch.object(workflow,'run_workflow') as run:
            self.assertEqual(cmd_pipeline(args),1)
            run.assert_not_called()
        media=self.root/'clip.wav';media.write_bytes(b'no model input')
        args=build_parser().parse_args(['pipeline',str(media),'--evidence-first','--translation-episode-context'])
        self.assertTrue(workflow.collect_jobs(args)[0].config.translation_episode_context)
        args=build_parser().parse_args(['pipeline','--translation-episode-context','--no-translation-episode-context'])
        self.assertFalse(args.translation_episode_context)

    def test_all_chosen_rows_exact_without_targets_evidence_or_entity_rewriting(self):
        original=[replace(b) for b in self.source]
        draft=[replace(b,text='OLD TARGET SENTINEL') for b in self.source]
        self.evidence[-1]=replace(self.evidence[-1],n='RIVAL ASR SENTINEL')
        with patch.object(quality,'call_llm',return_value='[1] 小雪。') as llm:
            rows=quality.query([1],self.source,self.evidence,self.config,quality.TRANSLATE,
                               'translation',self.cache,translated=draft)
        instruction=llm.call_args.args[1]
        document=self.context_document(instruction)
        self.assertEqual(document,[{'episode_row':b.index,'source':b.text} for b in self.source])
        self.assertNotIn('OLD TARGET SENTINEL',json.dumps(document))
        self.assertNotIn('RIVAL ASR SENTINEL',instruction)
        self.assertEqual(self.source,original)
        self.assertEqual(rows,{1:['小雪。']})

    def test_distant_source_change_and_flag_invalidate_only_draft_policy_cache(self):
        off=replace(self.config,translation_episode_context=False)
        with patch.object(quality,'call_llm',return_value='[1] 小雪。') as llm:
            for cfg in (off,self.config,self.config,off):
                quality.query([1],self.source,self.evidence,cfg,quality.TRANSLATE,'translation',self.cache)
            self.assertEqual(llm.call_count,2)
            changed=[replace(b) for b in self.source];changed[-1].text='遠くの村です。'
            quality.query([1],changed,self.evidence,self.config,quality.TRANSLATE,'translation',self.cache)
            self.assertEqual(llm.call_count,3)
        self.assertEqual(workflow.configuration_key(off),workflow.configuration_key(self.config))
        self.assertNotEqual(quality.draft_translation_settings(off),quality.draft_translation_settings(self.config))

    def test_toggle_reruns_draft_and_downstream_even_when_text_is_identical(self):
        media=self.root/'media.wav';media.write_bytes(b'untouched')
        source_path=self.root/'media.srt';translate.write_translated_srt(self.source,source_path)
        job=workflow.Job(media,source_path,'unit',replace(self.config,translation_episode_context=False))
        args=SimpleNamespace(force=False,proofread=False,review=False)
        decisions=[dict(line=b.index,status='accepted',candidate='W',text=b.text,reason='raw') for b in self.source]
        with patch.object(workflow,'build_evidence',return_value=self.evidence), \
             patch.object(workflow,'resolve_source',return_value=(self.source,decisions)) as resolve, \
             patch.object(workflow,'translate_scenes',return_value=[replace(b,text='测试译文。') for b in self.source]) as draft, \
             patch.object(workflow,'write_quality_result',wraps=quality.write_quality_result) as qa:
            workflow.process_language(job,args);workflow.process_language(job,args)
            self.assertEqual((resolve.call_count,draft.call_count,qa.call_count),(1,1,1))
            job.config=self.config
            workflow.process_language(job,args)
            self.assertEqual((resolve.call_count,draft.call_count,qa.call_count),(1,2,2))
        state=json.loads(job.state_path.read_text())
        self.assertTrue(state['translation']['request_settings']['episode_context']['enabled'])
        self.assertEqual(len(state['translation']['request_settings']['episode_context']['code_hash']),64)

    def test_missing_retry_keeps_full_source_and_rebases_only_requested_output_id(self):
        with patch.object(quality,'call_llm',side_effect=['[1] 小雪。','[1] 我很好。']) as llm:
            rows=quality.query([1,2],self.source,self.evidence,self.config,quality.TRANSLATE,'translation',self.cache)
        self.assertEqual(rows,{1:['小雪。'],2:['我很好。']})
        self.assertEqual(len(llm.call_args_list),2)
        for request in llm.call_args_list:
            self.assertEqual(len(self.context_document(request.args[1])),7)
        self.assertIn('本次仅处理这些ID: 1。',llm.call_args_list[1].args[0])

    def test_episode_row_number_cannot_replace_local_target_id(self):
        with patch.object(quality,'call_llm',return_value='[7] 遥远的小镇。'):
            rows=quality.query([7],self.source,self.evidence,self.config,quality.TRANSLATE,'translation',self.cache)
        self.assertEqual(rows,{})

    def test_other_stages_do_not_receive_episode_context_even_when_enabled(self):
        draft=[replace(b,text='测试译文。') for b in self.source]
        for kind,template,stage,answer in [('qa',quality.DIAGNOSE,'diagnosis','[1] OK'),
                ('patch',quality.PATCH,'targeted-repair','[1] FIX|小雪。'),
                ('resolve',quality.RESOLVE,'source-resolution','[1] KEEP|source agrees')]:
            with self.subTest(stage=stage),patch.object(quality,'call_llm',return_value=answer) as llm:
                quality.query([1],self.source,self.evidence,replace(self.config,stage=stage),
                              template,kind,self.root/f'{stage}.json',translated=draft)
                self.assertNotIn('READ-ONLY EPISODE SOURCE:',llm.call_args.args[1])

    def test_entity_target_contract_survives_original_unmarked_episode_context(self):
        replacement=EntityReplacement('E0001',0,2,'ユキ','⟦E0001⟧','g1','小雪','ユキ → 小雪')
        rows={b.index:EntityRowPlan(b.index,b.ts_line,b.text,
                 '⟦E0001⟧。' if b.index==1 else b.text,(replacement,) if b.index==1 else ()) for b in self.source}
        plan=EntityPlan([replace(b,text=rows[b.index].marked_source) for b in self.source],rows,[],{},
                       'context','key',{b.index:fingerprint(b.text) for b in self.source},'metadata')
        with patch.object(quality,'call_llm',return_value='[1] ⟦E0001⟧。') as llm:
            result=quality.query([1],self.source,self.evidence,self.config,quality.TRANSLATE,
                                 'translation',self.cache,entity_plan=plan)
        self.assertEqual(result,{1:['小雪。']})
        self.assertIn('⟦E0001⟧',llm.call_args.args[0])
        self.assertIn('exact marker',llm.call_args.args[1])
        self.assertEqual(self.context_document(llm.call_args.args[1])[0]['source'],'ユキ。')

    def test_oversized_or_missing_rows_fail_instead_of_truncating_before_model_call(self):
        huge=[replace(self.source[0],text='あ'*MAX_SOURCE_JSON_BYTES)]
        with patch.object(quality,'call_llm') as llm,self.assertRaisesRegex(ValueError,'no rows were truncated'):
            quality.translate_scenes(huge,[CueEvidence(1,huge[0].text,'A')],self.config,self.cache)
        llm.assert_not_called()
        for invalid in ([],[self.source[-1]],[self.source[0],self.source[0]],
                        [replace(self.source[0],text='')]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                episode_source_prefix(invalid)

    def test_length_limited_content_is_not_accepted_or_retried_without_context(self):
        cfg=replace(self.config,retries=2)
        answer=StreamResult(content='[1] 小雪。',finish_reason='length')
        with patch.object(translate,'_stream_response',return_value=answer) as stream, \
             patch.object(translate,'_record_request'),self.assertRaisesRegex(RuntimeError,'exhausted output/context'):
            translate.call_llm('source body','full context instruction',cfg)
        self.assertEqual(stream.call_count,1)
        with patch.object(translate,'_stream_response',return_value=answer),patch.object(translate,'_record_request'):
            self.assertEqual(translate.call_llm('source','instruction',replace(cfg,stage='diagnosis')),'[1] 小雪。')


if __name__=='__main__':
    unittest.main()
