"""Offline opt-in source-unit CLI and ASR resume regression checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser, cmd_pipeline
from src import workflow
from src.config import TranscribeConfig
from src.workflow_state import StageState


class SourcePauseCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.media=self.root/'clip.wav';self.media.write_bytes(b'unchanged local test media')
        self.output=self.root/'clip.srt'

    def args(self,*extra):
        return build_parser().parse_args(['pipeline',str(self.media),'--output',str(self.output),
            '--evidence-first','--arbitrate','-l','ja',*extra])

    def test_default_disabled_config_accepts_enabled_worker_option(self):
        self.assertFalse(self.args().source_pause_units)
        self.assertFalse(TranscribeConfig().source_pause_units)
        self.assertTrue(TranscribeConfig(source_pause_units=True).source_pause_units)
        self.assertTrue(self.args('--source-pause-units').source_pause_units)
        self.assertFalse(self.args('--source-pause-units','--no-source-pause-units').source_pause_units)

    def test_legacy_sequential_and_external_adjudication_rejected_before_context_or_models(self):
        legacy=build_parser().parse_args(['pipeline','--source-pause-units'])
        self.assertEqual(cmd_pipeline(legacy),1)
        adjudication=self.root/'saved.json';adjudication.write_text('[]')
        for extra in (('--asr-strategy','sequential'),('--adjudication',str(adjudication))):
            with patch.object(workflow,'prepare_context') as context, self.assertRaisesRegex(ValueError,'Source-pause'):
                workflow.run_workflow(self.args('--source-pause-units',*extra))
                context.assert_not_called()
        args=self.args('--source-pause-units');args.arbitrate=False
        with self.assertRaisesRegex(ValueError,'Source-pause'):
            workflow.run_workflow(args)

    def test_asr_cache_binds_flag_and_both_source_unit_dependencies_only_when_enabled(self):
        calls=[]
        def empty_worker(module,spec,directory,name):
            self.assertEqual(module,'src.ensemble_asr')
            calls.append(spec)
            # Config is sent directly to TranscribeConfig in the actual worker;
            # dependency hashes must not become unsupported model arguments.
            TranscribeConfig(**spec['config'])
            for job in spec['jobs']:
                StageState(Path(job['state'])).save('asr',job['key'],[],status='empty')
            return 0
        real_hash=workflow.file_hash
        revision={'source_units.py':'first','pause_layout.py':'first'}
        def hash_with_revision(path):
            path=Path(path)
            return revision[path.name] if path.name in revision else real_hash(path)
        with patch.object(workflow,'prepare_context'), patch.object(workflow,'run_worker',side_effect=empty_worker), \
             patch.object(workflow,'file_hash',side_effect=hash_with_revision):
            self.assertEqual(workflow.run_workflow(self.args()),0)
            self.assertEqual(workflow.run_workflow(self.args()),0)
            self.assertEqual(len(calls),1)
            self.assertNotIn('source_pause_units',calls[0]['config'])
            self.assertEqual(workflow.run_workflow(self.args('--source-pause-units')),0)
            self.assertTrue(calls[-1]['config']['source_pause_units'])
            self.assertEqual(workflow.run_workflow(self.args('--source-pause-units')),0)
            self.assertEqual(len(calls),2)
            revision['source_units.py']='second'
            workflow.run_workflow(self.args('--source-pause-units'))
            self.assertEqual(len(calls),3)
            revision['pause_layout.py']='second'
            workflow.run_workflow(self.args('--source-pause-units'))
            self.assertEqual(len(calls),4)
            workflow.run_workflow(self.args())
            self.assertEqual(len(calls),5)
            revision['source_units.py']='third';revision['pause_layout.py']='third'
            workflow.run_workflow(self.args())
            self.assertEqual(len(calls),5)


if __name__=='__main__':unittest.main()
