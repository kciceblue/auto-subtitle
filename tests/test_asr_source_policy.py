"""Offline source-preference, alignment-preservation, and Qwen context tests."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
from src import workflow
from src.asr_consensus import AudioWindow, aligned_tokens, choose_transcript, phrase_grade
from src.config import TranscribeConfig
from src.ensemble_asr import Recognizers, align_with_retry, select_window_source, transcribe_track
from src.translate import parse_srt


@dataclass
class AlignmentItem:
    text: str
    start_time: float
    end_time: float


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.addCleanup(self.temporary.cleanup)

    def test_original_q_preference_preserves_dissent_and_does_not_promote_grade(self):
        candidates = {'w': '明日は学校に行きます。', 'n': '明日は学校に行きます。',
                      'q': '明日は学校に行きません。'}
        before = dict(candidates)
        consensus = select_window_source(candidates, 'consensus')
        preferred = select_window_source(candidates, 'qwen')
        self.assertEqual((consensus['selected'], consensus['score']), choose_transcript(candidates))
        self.assertNotEqual(consensus['selected'], 'q')
        self.assertEqual(preferred['selected'], 'q')
        self.assertTrue(preferred['preferred_q'])
        self.assertIsNone(preferred['fallback_reason'])
        self.assertEqual(phrase_grade(candidates['q'], candidates, 'q'), 'C')
        self.assertEqual(candidates, before)
        only_q = {'q': candidates['q'], 'w': '', 'n': ''}
        self.assertEqual(select_window_source(only_q, 'qwen')['score'], 0)
        self.assertEqual(phrase_grade(only_q['q'], only_q, 'q'), 'C')

    def test_empty_or_visibly_repetitive_q_falls_back_with_reason(self):
        for text in ('', ' ', '。', 'ありがとう' * 10):
            with self.subTest(text=text):
                candidates = {'w': '明日会おう。', 'n': '明日会おう。', 'q': text}
                row = select_window_source(candidates, 'qwen')
                self.assertFalse(row['preferred_q'])
                self.assertTrue(row['fallback_reason'])
                self.assertEqual((row['selected'], row['score']), choose_transcript(candidates))
        with self.assertRaises(ValueError):
            select_window_source({'q': '明日会おう。'}, 'unknown')

    def test_complete_alignment_is_checked_before_overlap_ownership(self):
        window = AudioWindow(1, 9.2, 20.8, 10, 20)
        items = [dict(text='前', start_time=.1, end_time=.3),
                 dict(text='本当です', start_time=1, end_time=3),
                 dict(text='後', start_time=11, end_time=11.2)]
        result = aligned_tokens('前。本当です。後。', items, window, require_complete=True)
        self.assertEqual(''.join(row['text'] for row in result), '本当です。')
        source = '私は彼を助けます。'
        for words in ([], ['彼を助けます'], ['私は助けます'], ['私は彼を']):
            incomplete = [dict(text=word, start_time=1, end_time=2) for word in words]
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, 'complete selected transcript'):
                aligned_tokens(source, incomplete, AudioWindow(0, 0, 3, 0, 3), require_complete=True)
        valid = [dict(text='私は彼を助けます', start_time=1, end_time=2)]
        self.assertEqual(aligned_tokens(source, valid, AudioWindow(0, 0, 3, 0, 3),
                                        require_complete=True)[0]['text'], source)

    def test_locked_q_alignment_can_retry_same_text_once(self):
        q = '私は彼を助けます。'
        candidates = {'q': q, 'w': '彼を助けます。', 'n': '助けます。'}
        model = Mock()
        model.align.return_value = [SimpleNamespace(items=[AlignmentItem('私は彼を助けます', 1, 2)])]
        selected, tokens, failures = align_with_retry(model, [], 16000, AudioWindow(0, 0, 3, 0, 3),
                                                      candidates, 'q', [], lock_source=True)
        self.assertEqual(selected, 'q')
        self.assertEqual(''.join(t['text'] for t in tokens), q)
        self.assertEqual(model.align.call_count, 1)
        self.assertEqual(model.align.call_args.args[1], [q])
        self.assertEqual(failures[0]['model'], 'q')

    def test_locked_q_failure_never_tries_a_different_hypothesis(self):
        q = '私は彼を助けます。'
        model = Mock()
        model.align.return_value = [SimpleNamespace(items=[AlignmentItem('彼を助けます', 1, 2)])]
        with self.assertRaisesRegex(RuntimeError, 'Window 3: no valid forced alignment'):
            align_with_retry(model, [], 16000, AudioWindow(3, 0, 3, 0, 3),
                             {'q': q, 'w': '私は彼を助けたいと思っています。', 'n': '助けます。'},
                             'q', [], lock_source=True)
        self.assertEqual(model.align.call_count, 1)
        self.assertEqual(model.align.call_args.args[1], [q])

    def test_vocal_recovery_is_evidence_only_in_original_q_trial(self):
        original = {'w': '学校へ行きます。明日も行きます。',
                    'q': '今日は学校へ行きます。明日は休みです。', 'n': 'こんにちは。'}
        vocal = {key: '今日は学校へ行きます。明日は休みます。' for key in original}
        for policy in ('consensus', 'qwen'):
            directory = self.root / policy
            directory.mkdir()
            job = dict(media=str(directory / 'media.wav'), source=str(directory / 'episode.utterances.srt'),
                       audio=str(directory / 'audio.wav'), state=str(directory / 'episode.workflow.json'),
                       metadata=str(directory / 'episode.asr.json'),
                       adjudication=str(directory / 'episode.adjudication.json'),
                       hotwords=['学校'], key='test-key', media_hash='test-media')
            models = SimpleNamespace(batch_size=8)
            models.recognize = Mock(side_effect=[([dict(original)], {}), ([dict(vocal)], {})])
            def align(clips, texts, sr):
                return [SimpleNamespace(items=[AlignmentItem(text, .2, 1.8)]) for text in texts]
            models.align = Mock(side_effect=align)
            spec = dict(config=dict(no_demucs=False, ensemble_source=policy, qwen_asr_hotwords=False),
                        version='test', source_lang='Japanese', batch_size=8)
            with self.subTest(policy=policy), patch('src.audio.extract_audio'), \
                 patch('src.audio.separate_vocals', return_value=directory / 'vocals.wav'), \
                 patch('src.adjudicate.load_full_wav', return_value=(16000, np.ones(48000, dtype=np.float32))), \
                 patch('faster_whisper.vad.get_speech_timestamps', return_value=[dict(start=0, end=48000)]):
                transcribe_track(job, models, spec, directory)
            metadata = json.loads(Path(job['metadata']).read_text())
            recovered = metadata['recovery'][0]
            self.assertTrue(recovered['eligible_for_adoption'])
            self.assertEqual(metadata['raw_transcripts'], [original])
            self.assertEqual(metadata['ensemble_source'], policy)
            self.assertFalse(metadata['qwen_asr_hotwords'])
            self.assertIn('SDK-processed', metadata['qwen_text_provenance'])
            if policy == 'qwen':
                self.assertFalse(recovered['adopted'])
                self.assertTrue(recovered['policy_reason'])
                self.assertEqual(metadata['used_transcripts'], [original])
                self.assertTrue(metadata['source_selection'][0]['preferred_q'])
                self.assertEqual(metadata['source_selection'][0]['selected'], 'q')
                self.assertEqual(models.align.call_args.args[1], [original['q']])
                self.assertEqual(parse_srt(Path(job['source']))[0].text, original['q'])
                evidence = json.loads(Path(job['adjudication']).read_text())
                self.assertEqual(evidence[0]['source_model'], 'q')
                self.assertEqual(evidence[0]['grade'], 'C')
                self.assertNotEqual(evidence[0]['whisper'], evidence[0]['q'])
            else:
                self.assertTrue(recovered['adopted'])
                self.assertEqual(metadata['used_transcripts'], [vocal])
                self.assertFalse(metadata['source_selection'][0]['preferred_q'])

    def test_qwen_hotword_override_does_not_remove_whisper_bias(self):
        for enabled, words in ((True, ['名前', '学校']), (False, ['名前', '学校']), (True, [])):
            original_words = list(words)
            recognizers = Recognizers.__new__(Recognizers)
            recognizers.config = TranscribeConfig(language='ja', qwen_asr_hotwords=enabled)
            recognizers.batch_size = 8
            recognizers.w = SimpleNamespace(frames_per_second=50)
            recognizers.q = SimpleNamespace(transcribe_batch=Mock(return_value=['音声']))
            recognizers.n = SimpleNamespace(transcribe=Mock(return_value='音声'))
            pipeline = Mock()
            pipeline.transcribe.return_value = ([], None)
            with self.subTest(enabled=enabled), patch('faster_whisper.BatchedInferencePipeline', return_value=pipeline):
                recognizers.recognize(np.ones(16000, dtype=np.float32), 16000,
                                      [AudioWindow(0, 0, 1, 0, 1)], words)
            self.assertEqual(recognizers.q.transcribe_batch.call_args.kwargs['context'],
                             '、'.join(words) if enabled else '')
            self.assertEqual(pipeline.transcribe.call_args.kwargs['hotwords'], ' '.join(words) or None)
            self.assertEqual(words, original_words)

    def test_cli_defaults_and_inactive_ensemble_rejects_trial_controls(self):
        cfg = TranscribeConfig()
        self.assertEqual(cfg.ensemble_source, 'consensus')
        self.assertTrue(cfg.qwen_asr_hotwords)
        default = main.build_parser().parse_args(['pipeline'])
        self.assertEqual(default.ensemble_source, 'consensus')
        self.assertTrue(default.qwen_asr_hotwords)
        with self.assertRaises(ValueError):
            TranscribeConfig(ensemble_source='unknown')
        for controls in (['--ensemble-source', 'qwen'], ['--no-qwen-asr-hotwords']):
            args = main.build_parser().parse_args(['pipeline', '--evidence-first', *controls])
            with self.subTest(controls=controls), patch.object(workflow, 'prepare_context') as prepare:
                with self.assertRaisesRegex(ValueError, 'automatic --arbitrate'):
                    workflow.run_workflow(args)
                prepare.assert_not_called()
            args.evidence_first = False
            self.assertEqual(main.cmd_pipeline(args), 1)

    def test_language_trial_flags_cannot_be_silently_ignored_by_legacy_pipeline(self):
        for flag in ('--source-span-repair', '--entity-placeholders', '--delta-verification', '--structured-qa'):
            args = main.build_parser().parse_args(['pipeline', flag])
            with self.subTest(flag=flag):
                self.assertEqual(main.cmd_pipeline(args), 1)
        args = main.build_parser().parse_args(['pipeline', '--evidence-first', '--source-span-repair'])
        with patch.object(workflow, 'prepare_context') as prepare:
            with self.assertRaisesRegex(ValueError, 'automatic ensemble'):
                workflow.run_workflow(args)
            prepare.assert_not_called()

    def test_each_trial_flag_changes_worker_config_and_asr_fingerprint(self):
        media = self.root / 'input/work/track01.wav'
        media.parent.mkdir(parents=True)
        media.write_bytes(b'placeholder')
        args = main.build_parser().parse_args(['pipeline', '--evidence-first', '--arbitrate',
                                              '--no-auto-context', '-l', 'ja'])
        specs = []
        def prepare(jobs, args):
            for job in jobs:
                job.hotwords = ['名前']
        def worker(module, spec, directory, name):
            specs.append(spec)
            return 1
        with patch.object(workflow, 'prepare_context', side_effect=prepare), \
             patch.object(workflow, 'run_worker', side_effect=worker):
            for policy, hotwords in [('consensus', True), ('qwen', True), ('qwen', False)]:
                args.ensemble_source, args.qwen_asr_hotwords = policy, hotwords
                self.assertEqual(workflow.run_workflow(args), 1)
        self.assertEqual(len({s['jobs'][0]['key'] for s in specs}), 3)
        self.assertEqual([s['config']['ensemble_source'] for s in specs], ['consensus', 'qwen', 'qwen'])
        self.assertEqual([s['config']['qwen_asr_hotwords'] for s in specs], [True, True, False])
        self.assertEqual([s['jobs'][0]['hotwords'] for s in specs], [['名前']] * 3)


if __name__ == '__main__':
    unittest.main()
