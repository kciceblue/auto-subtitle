"""Offline NeMo admission and ASR cache identity contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser, cmd_pipeline
from src import workflow
from src.config import TranscribeConfig
from src.workflow_state import StageState, artifact_path, file_hash, fingerprint, write_json


class NemoCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root/'clip.wav'; self.media.write_bytes(b'local fixture')
        self.model = self.root/'model.nemo'; self.model.write_bytes(b'model fixture')
        self.python = self.root/'venv/bin/python'; self.python.parent.mkdir(parents=True)
        self.python.symlink_to(sys.executable)

    def args(self, *extra):
        return build_parser().parse_args(['pipeline',str(self.media),'-o',str(self.root/'clip.srt'),
            '--evidence-first','--arbitrate','-l','ja',*extra])

    def nemo_args(self, *extra):
        return self.args('--ensemble-source','nemo','--nemo-model',str(self.model),
                         '--nemo-python',str(self.python),*extra)

    def test_defaults_unchanged_and_interpreter_venv_path_preserved(self):
        self.assertEqual(self.args().ensemble_source,'consensus')
        self.assertIsNone(TranscribeConfig().nemo_model)
        config = TranscribeConfig(ensemble_source='nemo',nemo_model=self.model,nemo_python=self.python)
        self.assertEqual(config.nemo_python,self.python.absolute())
        self.assertNotEqual(config.nemo_python,self.python.resolve())

    def test_invalid_paths_and_unused_paths_fail_before_context_or_worker(self):
        missing = self.root/'missing.nemo'
        cases = [self.nemo_args('--nemo-model',str(missing)),
                 self.nemo_args('--nemo-python',str(self.model)),
                 self.args('--ensemble-source','nemo'),
                 self.args('--nemo-model',str(self.model))]
        for args in cases:
            with self.subTest(args=args):
                with patch.object(workflow,'prepare_context') as context, patch.object(workflow,'run_worker') as worker:
                    with self.assertRaises((ValueError,FileNotFoundError)):
                        workflow.run_workflow(args)
                    context.assert_not_called(); worker.assert_not_called()

    def test_incompatible_source_repairs_and_languages_rejected(self):
        for extra, message in [(('--source-span-repair',),'original Qwen ownership'),
                               (('-l','en'),'Japanese audio only'),
                               (('--asr-strategy','sequential'),'automatic --arbitrate')]:
            with self.subTest(extra=extra), patch.object(workflow,'prepare_context') as context:
                with self.assertRaisesRegex(ValueError,message):
                    workflow.run_workflow(self.nemo_args(*extra))
                context.assert_not_called()
        legacy=build_parser().parse_args(['pipeline','--ensemble-source','nemo'])
        self.assertEqual(cmd_pipeline(legacy),1)
        legacy_path=build_parser().parse_args(['pipeline','--nemo-model',str(self.model)])
        self.assertEqual(cmd_pipeline(legacy_path),1)

    def test_runtime_model_and_code_identity_invalidate_asr_cache(self):
        calls=[]
        identity={'model_sha256':'model-a','environment':'env-a','bridge':'code-a'}
        def empty_worker(module,spec,directory,name):
            self.assertEqual(module,'src.ensemble_asr')
            TranscribeConfig(**spec['config'])
            self.assertEqual(spec['nemo_runtime_identity'],identity)
            calls.append(spec)
            for job in spec['jobs']:
                StageState(Path(job['state'])).save('asr',job['key'],[],status='empty')
        consumer={'revision':'ensemble-a'}
        real_hash=workflow.file_hash
        def consumer_hash(path):
            return consumer['revision'] if Path(path).name=='ensemble_asr.py' else real_hash(path)
        with patch.object(workflow,'file_hash',side_effect=consumer_hash), \
             patch('src.nemo_asr.runtime_identity',side_effect=lambda cfg: dict(identity)), \
             patch.object(workflow,'prepare_context'), patch.object(workflow,'run_worker',side_effect=empty_worker):
            workflow.run_workflow(self.nemo_args()); workflow.run_workflow(self.nemo_args())
            self.assertEqual(len(calls),1)
            for key in identity:
                identity[key]+='-changed'
                workflow.run_workflow(self.nemo_args())
                self.assertEqual(len(calls),1+list(identity).index(key)+1)
                workflow.run_workflow(self.nemo_args())
                self.assertEqual(len(calls),1+list(identity).index(key)+1)
            consumer['revision']='ensemble-b'
            workflow.run_workflow(self.nemo_args())
            self.assertEqual(len(calls),5)

    def test_nonempty_nemo_asr_resumes_and_tampered_prepass_forces_worker(self):
        calls=[]
        prepasses=[]
        def worker(module,spec,directory,name):
            self.assertEqual(module,'src.ensemble_asr')
            calls.append(spec)
            for job in spec['jobs']:
                source=Path(job['source']); source.parent.mkdir(parents=True,exist_ok=True)
                source.write_text('1\n00:00:01,000 --> 00:00:02,000\n明日会おう\n',encoding='utf-8')
                metadata=Path(job['metadata']); write_json(metadata,{'source_model':'r'})
                prepass=artifact_path(source,'.nemo-prepass.json')
                write_json(prepass,{'texts':['明日会おう'],'complete':True})
                prepasses.append(prepass)
                adjudication=Path(job['adjudication'])
                write_json(adjudication,[{'line':1,'w':'明日会おう','r':'明日会おう','grade':'C'}])
                state=StageState(Path(job['state']))
                state.save('asr',job['key'],[source,metadata,prepass])
                key=fingerprint([spec['version'],job['media_hash'],file_hash(source),
                                 spec['source_lang'],spec['batch_size']])
                state.save('arbitration',key,[adjudication])
            return 0
        def language(job,args):job.successful=True
        with patch('src.nemo_asr.runtime_identity',return_value={'model':'frozen'}), \
             patch.object(workflow,'prepare_context'), \
             patch.object(workflow,'run_worker',side_effect=worker), \
             patch.object(workflow,'process_language',side_effect=language) as translate:
            self.assertEqual(workflow.run_workflow(self.nemo_args()),0)
            self.assertEqual((len(calls),translate.call_count),(1,1))
            self.assertEqual(workflow.run_workflow(self.nemo_args()),0)
            self.assertEqual((len(calls),translate.call_count),(1,2))
            prepasses[-1].write_text('tampered evidence',encoding='utf-8')
            self.assertEqual(workflow.run_workflow(self.nemo_args()),0)
            self.assertEqual((len(calls),translate.call_count),(2,3))
            prepasses[-1].unlink()
            self.assertEqual(workflow.run_workflow(self.nemo_args()),0)
            self.assertEqual((len(calls),translate.call_count),(3,4))

if __name__=='__main__': unittest.main()
