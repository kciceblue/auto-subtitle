"""Offline NeMo source identity, strict alignment and isolated prepass regressions."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.asr import Segment
from src.asr_consensus import AudioWindow, aligned_tokens
from src.config import TranslateConfig
from src.ensemble_asr import (align_with_retry, audio_window_identity, merge_window_utterances,
    prepare_nemo_track, read_nemo_track, run, select_window_source, transcribe_track)
from src.evidence import build_evidence
from src.nemo_asr import NemoASRError
from src.translate import parse_srt
from src.workflow_state import artifact_path, file_hash, write_json


@dataclass
class Item:
    text: str
    start_time: float
    end_time: float


class NemoEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sr = 16000
        self.audio = np.ones(3 * self.sr, dtype=np.float32)
        self.windows = [AudioWindow(0, 0, 3, 0, 3)]
        self.raw = {'w': '明日は学校に行きます。', 'n': '明日は学校に行きます',
                    'q': '明日は学校に行きません。'}
        self.job = dict(media=str(self.root/'media.wav'), source=str(self.root/'episode.utterances.srt'),
            audio=str(self.root/'audio.wav'), state=str(self.root/'episode.workflow.json'),
            metadata=str(self.root/'episode.asr.json'), adjudication=str(self.root/'episode.adjudication.json'),
            hotwords=[], key='test-key', media_hash='test-media')
        self.spec = dict(config=dict(no_demucs=True, ensemble_source='nemo', qwen_asr_hotwords=False),
                         version='test', source_lang='Japanese', batch_size=8,
                         nemo_runtime_identity={'model': 'frozen-model'})

    def binding(self, text):
        path = self.root/'staged.json'
        write_json(path, dict(input=audio_window_identity(self.audio, self.sr, self.windows),
            texts=[text], report={'windows':[{'text':text,'raw_return':{'text':text}}]}, seconds=2.5))
        return dict(path=str(path), sha256=file_hash(path))

    def test_nemo_is_selected_exactly_but_never_an_extra_vote(self):
        candidates = dict(self.raw, r='明日は学校に行くつもりです')
        old = deepcopy(candidates)
        selected = select_window_source(candidates, 'nemo')
        self.assertEqual(selected['selected'], 'r')
        self.assertTrue(selected['preferred_r'])
        self.assertTrue(selected['lock_source'])
        self.assertFalse(selected['preferred_q'])
        self.assertEqual(selected['score'], 0.0)
        self.assertEqual(candidates, old)
        for policy in ['consensus', 'qwen']:
            self.assertEqual(select_window_source(candidates, policy), select_window_source(self.raw, policy))
        self.assertIsNone(selected['fallback_reason'])

    def test_unusable_nemo_explicitly_falls_back_through_q_preference(self):
        for r in ['', '。', 'ありがとう'*8]:
            selected = select_window_source(dict(self.raw, r=r), 'nemo')
            self.assertEqual(selected['selected'], 'q')
            self.assertTrue(selected['preferred_q'])
            self.assertFalse(selected['preferred_r'])
            self.assertTrue(selected['lock_source'])
            self.assertIn('NeMo', selected['fallback_reason'])
        empty = dict(self.raw, q='', r='')
        selected = select_window_source(empty, 'nemo')
        self.assertEqual(selected['selected'], select_window_source(empty, 'qwen')['selected'])
        self.assertIn('consensus', selected['fallback_reason'])
        self.assertFalse(selected['lock_source'])

    def test_locked_nemo_missing_suffix_never_switches_to_q(self):
        text = '私は彼を助けるつもりです'
        models = Mock()
        models.align.return_value = [SimpleNamespace(items=[Item('私は彼を助ける', .2, 2.)])]
        with self.assertRaisesRegex(RuntimeError, 'no valid forced alignment'):
            align_with_retry(models, self.audio, self.sr, self.windows[0],
                dict(self.raw, r=text), 'r', [], lock_source=True)
        self.assertEqual(models.align.call_count, 1)
        self.assertEqual(models.align.call_args.args[1], [text])
        models.align.return_value = [SimpleNamespace(items=[Item(text, .2, 2.)])]
        chosen, tokens, _ = align_with_retry(models, self.audio, self.sr, self.windows[0],
            dict(self.raw, r=text), 'r', [], lock_source=True)
        self.assertEqual(chosen, 'r')
        self.assertEqual(''.join(t['text'] for t in tokens), text)

    def test_staged_samples_windows_and_report_cannot_be_stale(self):
        binding = self.binding('明日会おう')
        self.assertEqual(read_nemo_track(binding, self.audio, self.sr, self.windows)['texts'], ['明日会おう'])
        changed = self.audio.copy(); changed[400] = 0.5
        for audio, windows in [(changed,self.windows), (self.audio,[AudioWindow(0,0,3,.1,3)])]:
            with self.assertRaisesRegex(ValueError, 'audio/window identity mismatch'):
                read_nemo_track(binding, audio, self.sr, windows)
        Path(binding['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            read_nemo_track(binding, self.audio, self.sr, self.windows)

    def test_prepass_freezes_runtime_identity_and_keeps_raw_report(self):
        text='明日会おう'; report={'windows':[{'text':text,'raw_return':{'text':text,'score':-1.}}]}
        with patch('src.ensemble_asr.load_track_windows', return_value=(Path(self.job['audio']),self.sr,self.audio,[],self.windows)), \
             patch('src.nemo_asr.transcribe_windows', return_value=([text],report)) as decode:
            result = prepare_nemo_track(self.job,self.spec,self.root/'prepass.json')
        self.assertEqual(decode.call_args.kwargs['expected_identity'],self.spec['nemo_runtime_identity'])
        saved=read_nemo_track(result,self.audio,self.sr,self.windows)
        self.assertEqual(saved['report'], report)
        self.assertEqual(saved['texts'],[text])
        self.assertEqual(Path(result['path']).stat().st_mode & 0o222,0)
        self.assertIn('before W/Q/N',saved['execution'])

    def test_failed_prepass_preserves_partial_report_and_loads_no_gpu_models(self):
        report={'complete':False,'windows':[{'text':'partial raw','raw_return':{'text':'partial raw'}}]}
        with patch('src.ensemble_asr.load_track_windows', return_value=(Path(self.job['audio']),self.sr,self.audio,[],self.windows)), \
             patch('src.nemo_asr.transcribe_windows',side_effect=NemoASRError('worker failed',report)), \
             patch('src.ensemble_asr.Recognizers') as models:
            self.assertEqual(run(dict(self.spec,jobs=[self.job])),1)
        models.assert_not_called()
        saved=json.loads(artifact_path(Path(self.job['source']),'.nemo-prepass.json').read_text())
        self.assertEqual(saved['report'],report)
        self.assertFalse(saved['complete'])
        self.assertFalse(Path(self.job['source']).exists())

    def test_every_job_prepass_finishes_before_resident_models_load(self):
        events=[]
        jobs=[dict(self.job,source=str(self.root/f'episode{i}.srt')) for i in range(2)]
        def prepare(job,spec,path):
            events.append(('prepass',job['source']));return {'path':str(path),'sha256':'test'}
        def load(config,batch):events.append(('models',None));return object()
        def track(job,models,spec,temporary,**kwargs):events.append(('track',job['source']))
        with patch('src.ensemble_asr.prepare_nemo_track',side_effect=prepare), \
             patch('src.ensemble_asr.Recognizers',side_effect=load), \
             patch('src.ensemble_asr.transcribe_track',side_effect=track):
            self.assertEqual(run(dict(self.spec,jobs=jobs)),0)
        self.assertEqual([event[0] for event in events],['prepass','prepass','models','track','track'])

    def test_track_preserves_raw_r_and_wqn_without_punctuation_or_approval(self):
        text='明日は学校に行きます明後日も行きます'
        original={k:text for k in ['w','q','n']}
        original['q']='明日は学校に行きます。明後日も行きます。'
        models=SimpleNamespace(batch_size=8,recognize=Mock(return_value=([original],{})))
        models.align=Mock(side_effect=lambda clips,texts,sr:[SimpleNamespace(items=[Item(t,.2,2.)]) for t in texts])
        with patch('src.ensemble_asr.load_track_windows',return_value=(Path(self.job['audio']),self.sr,self.audio,[],self.windows)), \
             patch('src.ensemble_asr.restore_sentence_boundaries') as punctuate:
            transcribe_track(self.job,models,self.spec,self.root,nemo_prepass=self.binding(text))
        punctuate.assert_not_called()
        metadata=json.loads(Path(self.job['metadata']).read_text())
        self.assertEqual(metadata['raw_transcripts'],[dict(original,r=text)])
        self.assertEqual(metadata['used_transcripts'],[dict(original,r=text)])
        self.assertEqual(metadata['segments'][0]['text'],text)
        self.assertFalse(metadata['nemo_independent_vote'])
        self.assertEqual(metadata['nemo_prepass']['seconds'],2.5)
        source=parse_srt(Path(self.job['source']))
        evidence=build_evidence(source,TranslateConfig(adjudication=Path(self.job['adjudication'])))
        self.assertEqual(evidence[0].source_model,'r')
        self.assertEqual(evidence[0].raw,text)
        self.assertEqual(evidence[0].r,text)
        self.assertEqual(evidence[0].grade,'C')
        self.assertTrue(evidence[0].blocked)
        self.assertEqual(evidence[0].candidates()['R'],text)
        self.assertIn('Reazon', '\n'.join(evidence[0].notes()))
        self.assertEqual(models.align.call_args.args[1],[text])

    def test_merged_nemo_preserves_punctuation_tokens_and_model_identity(self):
        segments=[Segment(1,1,2,'古い手紙。'),Segment(2,2.5,3.5,'を見つけた')]
        rows=[dict(line=i+1,window=i,w=s.text,whisper='独立W',n='独立N',q='独立Q',r=s.text,
                   source_model='r',source_policy='nemo',grade='C',note='raw') for i,s in enumerate(segments)]
        tokens=[[dict(text=s.text,start=s.start,end=s.end)] for s in segments]
        old=deepcopy(tokens)
        merged,evidence,words=merge_window_utterances(segments,rows,tokens)
        self.assertEqual(merged[0].text,'古い手紙。を見つけた')
        self.assertEqual(words[0],old[0]+old[1])
        self.assertEqual(evidence[0]['source_model'],'r')
        self.assertEqual(evidence[0]['r'],merged[0].text)
        self.assertEqual(evidence[0]['windows'],[0,1])
        self.assertEqual(evidence[0]['grade'],'C')

    def test_unpunctuated_nemo_windows_do_not_chain_into_mega_units(self):
        segments=[Segment(i+1,i*20.,i*20.+19.,text) for i,text in enumerate(
            ['今日は晴れです','明日は学校へ行きます','週末は家にいます'])]
        rows=[dict(line=i+1,window=i,w=s.text,whisper=s.text,n=s.text,q=s.text,r=s.text,
                   source_model='r',source_policy='nemo',grade='C',note='raw') for i,s in enumerate(segments)]
        tokens=[[dict(text=s.text,start=s.start,end=s.end)] for s in segments]
        before=deepcopy((segments,tokens))
        merged,evidence,words=merge_window_utterances(segments,rows,tokens)
        self.assertEqual(merged,before[0])
        self.assertEqual(words,before[1])
        self.assertEqual([row['windows'] for row in evidence],[[0],[1],[2]])
        self.assertEqual([row['source_model'] for row in evidence],['r']*3)

    def test_nemo_explicit_continuation_cannot_exceed_combined_span_limit(self):
        for final_end,expected_count in [(30.,1),(30.001,2)]:
            segments=[Segment(1,0.,19.,'古い手紙'),Segment(2,19.5,final_end,'を見つけた')]
            rows=[dict(line=i+1,window=i,w=s.text,whisper=s.text,n=s.text,q=s.text,r=s.text,
                       source_model='r',source_policy='nemo',grade='C',note='raw') for i,s in enumerate(segments)]
            tokens=[[dict(text=s.text,start=s.start,end=s.end)] for s in segments]
            before=deepcopy(tokens)
            merged,evidence,words=merge_window_utterances(segments,rows,tokens)
            self.assertEqual(len(merged),expected_count)
            self.assertEqual(''.join(s.text for s in merged),'古い手紙を見つけた')
            self.assertEqual([t for group in words for t in group],[t for group in before for t in group])
            self.assertTrue(all(row['grade']=='C' for row in evidence))

    def test_prepass_retries_preserve_every_prior_success_and_failure_exactly(self):
        path=self.root/'prepass.json'
        failed={'complete':False,'windows':[{'text':'partial raw'}]}
        results=[(['最初の音声'],{'complete':True,'raw':['最初の音声']}),
                 NemoASRError('partial failure',failed),
                 (['最後の音声'],{'complete':True,'raw':['最後の音声']})]
        with patch('src.ensemble_asr.load_track_windows',return_value=(Path(self.job['audio']),self.sr,self.audio,[],self.windows)), \
             patch('src.nemo_asr.transcribe_windows',side_effect=results):
            prepare_nemo_track(self.job,self.spec,path)
            first=path.read_bytes(); first_hash=file_hash(path)
            with self.assertRaises(NemoASRError):prepare_nemo_track(self.job,self.spec,path)
            second=path.read_bytes(); second_hash=file_hash(path)
            self.assertEqual(json.loads(second)['report'],failed)
            prepare_nemo_track(self.job,self.spec,path)
        for digest,original in [(first_hash,first),(second_hash,second)]:
            snapshot=path.with_name(f'{path.stem}.attempt-{digest}{path.suffix}')
            self.assertEqual(snapshot.read_bytes(),original)
            self.assertEqual(snapshot.stat().st_mode & 0o222,0)
        self.assertEqual(json.loads(path.read_text())['texts'],['最後の音声'])

    def test_prepass_snapshot_failure_aborts_before_decoder_or_overwrite(self):
        path=self.root/'prepass.json'; path.write_bytes(b'original evidence')
        with patch('src.translate.make_snapshot',side_effect=OSError('snapshot unavailable')), \
             patch('src.nemo_asr.transcribe_windows') as decode:
            with self.assertRaisesRegex(OSError,'snapshot unavailable'):
                prepare_nemo_track(self.job,self.spec,path)
        decode.assert_not_called()
        self.assertEqual(path.read_bytes(),b'original evidence')

    def test_corrupt_existing_prepass_snapshot_never_allows_overwrite(self):
        path=self.root/'prepass.json'; path.write_bytes(b'original evidence')
        digest=file_hash(path)
        snapshot=path.with_name(f'{path.stem}.attempt-{digest}{path.suffix}')
        snapshot.write_bytes(b'different evidence')
        with patch('src.nemo_asr.transcribe_windows') as decode:
            with self.assertRaisesRegex(RuntimeError,'snapshot differs'):
                prepare_nemo_track(self.job,self.spec,path)
        decode.assert_not_called()
        self.assertEqual(path.read_bytes(),b'original evidence')


if __name__=='__main__':unittest.main()
