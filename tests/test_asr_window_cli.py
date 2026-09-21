"""ASR context window geometry and opt-in cache/admission contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from main import build_parser,cmd_pipeline
from src import workflow
from src.asr_consensus import audio_windows
from src.config import TranscribeConfig
from src.workflow_state import StageState

class AsrWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.media=self.root/'local.wav';self.media.write_bytes(b'fixture')
    def args(self,*extra):
        return build_parser().parse_args(['pipeline',str(self.media),'-o',str(self.root/'episode.srt'),
            '--evidence-first','--arbitrate','-l','ja',*extra])
    def test_defaults_and_range(self):
        self.assertEqual(TranscribeConfig().asr_window_seconds,20.0)
        self.assertEqual(self.args().asr_window_seconds,20.0)
        for value in [True,None,7.99,28.01,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                TranscribeConfig(asr_window_seconds=value)
        self.assertEqual(TranscribeConfig(asr_window_seconds=28.).asr_window_seconds,28.)
    def test_complete_timeline_and_sub30_second_context(self):
        for maximum in [8.,20.,28.]:
            for duration in [1.,28.,80.5,1420.178866]:
                rows=audio_windows(duration,[],maximum=maximum)
                self.assertEqual(rows[0].core_start,0.)
                self.assertEqual(rows[-1].core_end,duration)
                self.assertTrue(all(a.core_end==b.core_start for a,b in zip(rows,rows[1:])))
                self.assertTrue(all(0<=r.start<=r.core_start<r.core_end<=r.end<=duration for r in rows))
                self.assertTrue(all(r.end-r.start<=29.600001 for r in rows))
        self.assertLess(len(audio_windows(1420.,[],maximum=28.)),len(audio_windows(1420.,[])))
    def test_invalid_or_inapplicable_fails_before_context_models(self):
        for extra in [('--asr-window-seconds','nan'),('--asr-window-seconds','29'),
                      ('--asr-window-seconds','28','--asr-strategy','sequential')]:
            with patch.object(workflow,'prepare_context') as context,patch.object(workflow,'run_worker') as worker:
                with self.assertRaises(ValueError):workflow.run_workflow(self.args(*extra))
                context.assert_not_called();worker.assert_not_called()
        args=build_parser().parse_args(['pipeline','--asr-window-seconds','28'])
        self.assertEqual(cmd_pipeline(args),1)
    def test_window_and_code_change_invalidate_asr_cache(self):
        calls=[];revision=['initial'];real_hash=workflow.file_hash
        def hash_path(path):
            return revision[0] if Path(path).name=='asr_consensus.py' else real_hash(path)
        def worker(module,spec,*unused):
            TranscribeConfig(**spec['config']);calls.append(spec)
            for job in spec['jobs']:StageState(Path(job['state'])).save('asr',job['key'],[],status='empty')
        with patch.object(workflow,'file_hash',side_effect=hash_path),patch.object(workflow,'prepare_context'),patch.object(workflow,'run_worker',side_effect=worker):
            workflow.run_workflow(self.args());workflow.run_workflow(self.args());self.assertEqual(len(calls),1)
            self.assertNotIn('asr_window_seconds',calls[0]['config'])
            workflow.run_workflow(self.args('--asr-window-seconds','28'));workflow.run_workflow(self.args('--asr-window-seconds','28'))
            self.assertEqual(len(calls),2);self.assertEqual(calls[-1]['config']['asr_window_seconds'],28.)
            revision[0]='changed';workflow.run_workflow(self.args('--asr-window-seconds','28'));self.assertEqual(len(calls),3)
            workflow.run_workflow(self.args());self.assertEqual(len(calls),4)
            revision[0]='again';workflow.run_workflow(self.args());self.assertEqual(len(calls),4)

if __name__=='__main__':unittest.main()
