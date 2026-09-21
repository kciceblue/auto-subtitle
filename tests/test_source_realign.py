"""Forced-realignment isolation, exact coverage and conservative timing guards."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.source_realign import (_launch_worker, _validate_items, realign_changed,
                                VERSION, _worker)
from src.translate import SrtBlock
from src.workflow_state import fingerprint, read_json, write_json


def items(start=.32, end=2.32):
    return [{'text': '新しい', 'start_time': start, 'end_time': 1.0},
            {'text': '言葉', 'start_time': 1.4, 'end_time': end}]


def validate(value=None, text='新しい言葉。', **kwargs):
    args = dict(source_id=1, core_start=1., core_end=4., clip_start=.68, clip_end=4.32)
    args.update(kwargs)
    return _validate_items(text, items() if value is None else value, **args)


class AlignmentValidationTests(unittest.TestCase):
    def test_exact_text_and_real_pause_are_preserved(self):
        tokens = validate(text=' 新しい 言葉。 ')
        self.assertEqual(''.join(t['text'] for t in tokens), ' 新しい 言葉。 ')
        self.assertAlmostEqual(tokens[1]['start']-tokens[0]['end'], .4)
        self.assertEqual(tokens[0]['begin'], 0)
        self.assertEqual(tokens[-1]['finish'], len(' 新しい 言葉。 '))
        self.assertTrue(all(t['offset_scope']=='utterance' and t['source_id']==1 for t in tokens))

    def test_tiny_zero_bin_at_existing_boundary_shares_interval_without_new_time(self):
        raw = [{'text':'今日','start_time':.32,'end_time':1.0},
               {'text':'は','start_time':1.0,'end_time':1.0},
               {'text':'晴れ','start_time':1.4,'end_time':2.32}]
        before = deepcopy(raw)
        tokens = validate(raw,text='今日は晴れ。')
        self.assertEqual(raw,before)
        self.assertEqual(len(tokens),2)
        self.assertEqual(tokens[0]['text'],'今日は')
        self.assertAlmostEqual(tokens[0]['end'],1.68)
        self.assertEqual(tokens[0]['alignment_grain'],'group')
        self.assertEqual(len(tokens[0]['alignment_group_items']),2)
        self.assertEqual(''.join(t['text'] for t in tokens),'今日は晴れ。')

    def test_zero_bin_without_shared_boundary_or_excessive_text_stays_invalid(self):
        for text, zeros in [('今日なら晴れ。',[{'text':'なら','start_time':1.2,'end_time':1.2}]),
                            ('今日はもう晴れ。',[{'text':'はもう','start_time':1.,'end_time':1.}])]:
            raw = [{'text':'今日','start_time':.32,'end_time':1.},*zeros,
                   {'text':'晴れ','start_time':1.4,'end_time':2.32}]
            with self.subTest(text=text),self.assertRaises(ValueError):
                validate(raw,text=text)

    def test_missing_content_never_becomes_complete_alignment(self):
        with self.assertRaisesRegex(ValueError, 'complete'):
            validate([{'text': '来', 'start_time': .32, 'end_time': 1}], text='来ない。')

    def test_collapsed_nonfinite_reversed_and_overlapping_tokens_fail(self):
        variants=[]
        for start,end in [(1,1),(float('nan'),2),(1,float('inf')),(2,1),(-.001,1)]:
            row=items();row[0].update(start_time=start,end_time=end);variants.append(row)
        row=items();row[1]['start_time']=.99;variants.append(row)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):validate(value)

    def test_only_bounded_edge_jitter_is_clamped(self):
        tokens=validate(items(start=.28,end=3.36))
        self.assertAlmostEqual(tokens[0]['start'],1.)
        self.assertAlmostEqual(tokens[-1]['end'],4.)
        with self.assertRaisesRegex(ValueError,'edge jitter'):
            validate(items(start=.1))

    def test_clamping_never_invents_an_80ms_word(self):
        value=[{'text':'新しい','start_time':.25,'end_time':.32},
               {'text':'言葉','start_time':1.4,'end_time':2.32}]
        with self.assertRaisesRegex(ValueError,'collapse'):validate(value)

    def test_outside_crop_or_oversized_token_is_rejected(self):
        with self.assertRaises(ValueError):validate(items(end=5.0))
        with self.assertRaisesRegex(ValueError,'long aligned token'):
            _validate_items('言葉。',[{'text':'言葉','start_time':0,'end_time':5}],
                source_id=1,core_start=0,core_end=10,clip_start=0,clip_end=10)


class RealignmentParentTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name);self.media=self.directory/'original.wav';self.media.write_bytes(b'unchanged media')
        self.output=self.directory/'episode.source-realignment.json'
        self.source=[SrtBlock(1,'00:00:01,000 --> 00:00:04,000','新しい言葉。'),
                     SrtBlock(2,'00:00:05,000 --> 00:00:08,000','そのまま。')]
        self.original={'raw_transcripts':[{'w':'old','n':'old','q':'old'}],
          'used_transcripts':[{'q':'old'}], 'utterance_tokens':[
            [{'text':'古い言葉。','start':1.,'end':4.,'begin':10,'finish':15}],
            [{'text':'そのまま。','start':5.,'end':8.,'begin':15,'finish':20}]],
          'segments':[{'index':1,'start':1.,'end':4.,'text':'古い言葉。'},
                      {'index':2,'start':5.,'end':8.,'text':'そのまま。'}]}

    def response(self,spec,timeout):
        return {'request_key':spec['request_key'],'rows':{
            str(job['id']):{'source_hash':job['source_hash'],'clip_start':job['core_start']-.32,
                           'clip_end':job['core_end']+.32,'items':items()}
            for job in spec['jobs']},'peak_cuda_allocated_bytes':123}

    def test_only_successful_token_row_changes_and_raw_is_preserved(self):
        before=deepcopy((self.source,self.original));media=self.media.read_bytes()
        with patch('src.source_realign._launch_worker',side_effect=self.response) as worker:
            result=realign_changed(self.media,self.source,[1],self.original,output_path=self.output)
        self.assertEqual(result.successful_ids,(1,));self.assertFalse(result.failed_ids)
        self.assertEqual((self.source,self.original),before);self.assertEqual(self.media.read_bytes(),media)
        self.assertEqual(result.metadata['utterance_tokens'][1],self.original['utterance_tokens'][1])
        for key in self.original:
            if key!='utterance_tokens':self.assertEqual(result.metadata[key],self.original[key])
        self.assertEqual(''.join(t['text'] for t in result.metadata['utterance_tokens'][0]),self.source[0].text)
        self.assertEqual(read_json(self.output)['original_metadata'],self.original)
        self.assertEqual(worker.call_args.args[0]['media'],str(self.media.resolve()))

    def test_failed_row_keeps_original_tokens_while_other_success_survives(self):
        def response(spec,timeout):
            result=self.response(spec,timeout)
            result['rows']['2']={'source_hash':spec['jobs'][1]['source_hash'],'error':'no coverage'}
            return result
        with patch('src.source_realign._launch_worker',side_effect=response):
            result=realign_changed(self.media,self.source,[1,2],self.original)
        self.assertEqual(result.successful_ids,(1,));self.assertEqual(result.failed_ids,{2:'no coverage'})
        self.assertEqual(result.metadata['utterance_tokens'][1],self.original['utterance_tokens'][1])

    def test_parent_revalidates_worker_timing_and_hashes(self):
        for failure in ['collapsed','stale_hash','extra_id','stale_request']:
            def response(spec,timeout):
                result=self.response(spec,timeout)
                if failure=='collapsed':result['rows']['1']['items'][0]['end_time']=.32
                elif failure=='stale_hash':result['rows']['1']['source_hash']='different'
                elif failure=='extra_id':result['rows']['999']=dict(result['rows']['1'])
                else:result['request_key']='different'
                return result
            with self.subTest(failure=failure),patch('src.source_realign._launch_worker',side_effect=response):
                result=realign_changed(self.media,self.source,[1],self.original)
            self.assertFalse(result.successful_ids);self.assertIn(1,result.failed_ids)
            self.assertEqual(result.metadata,self.original)

    def test_worker_timeout_is_explicit_and_leaves_all_original_tokens(self):
        with patch('src.source_realign._launch_worker',side_effect=subprocess.TimeoutExpired(['worker'],3)):
            result=realign_changed(self.media,self.source,[1],self.original)
        self.assertIn(1,result.failed_ids);self.assertEqual(result.metadata,self.original)

    def test_changed_original_interval_is_rejected_before_model_launch(self):
        source=deepcopy(self.source);source[0].ts_line='00:00:02,000 --> 00:00:04,000'
        with patch('src.source_realign._launch_worker') as worker:
            result=realign_changed(self.media,source,[1],self.original)
        worker.assert_not_called();self.assertIn(1,result.failed_ids)

    def test_empty_change_set_does_not_launch_cuda_worker(self):
        with patch('src.source_realign._launch_worker') as worker:
            result=realign_changed(self.media,self.source,[],self.original)
        worker.assert_not_called();self.assertFalse(result.failed_ids);self.assertEqual(result.metadata,self.original)

    def test_changed_media_invalidates_all_successful_rows(self):
        def response(spec,timeout):
            result=self.response(spec,timeout);self.media.write_bytes(b'changed while running');return result
        with patch('src.source_realign._launch_worker',side_effect=response):
            result=realign_changed(self.media,self.source,[1],self.original)
        self.assertFalse(result.successful_ids);self.assertEqual(result.metadata,self.original)
        self.assertIn('media changed',result.failed_ids[1])

    def test_sidecar_cannot_overwrite_raw_asr_or_source_artifact(self):
        for name in ['episode.asr.json','source.srt']:
            with self.assertRaises(ValueError):
                realign_changed(self.media,self.source,[1],self.original,output_path=self.directory/name)
        with self.assertRaises(ValueError):
            realign_changed(self.media,self.source,[1,1],self.original)

    def test_worker_crops_original_audio_and_loads_only_one_aligner(self):
        import numpy as np
        from src.workflow_state import file_hash
        @dataclass
        class AlignedItem:
            text: str
            start_time: float
            end_time: float
        audio = np.linspace(-1, 1, 160000, dtype=np.float32)
        model = Mock()
        model.align.return_value = [SimpleNamespace(items=[AlignedItem(**r) for r in items()])]
        fake_torch = SimpleNamespace(inference_mode=nullcontext,
            cuda=SimpleNamespace(max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200))
        spec = {'request_key': 'key', 'media': str(self.media), 'media_hash': file_hash(self.media),
                'aligner_model': 'local-aligner', 'padding_seconds': .32, 'language': 'Japanese',
                'jobs': [{'id': 1, 'text': self.source[0].text, 'source_hash': fingerprint(self.source[0].text),
                          'core_start': 1., 'core_end': 4.}]}
        with patch('src.adjudicate.load_full_wav', return_value=(16000, audio)) as load_audio, \
             patch('src.source_realign._load_aligner', return_value=(model, 300)) as load_model, \
             patch.dict('sys.modules', {'torch': fake_torch}):
            result = _worker(spec)
        load_audio.assert_called_once_with(self.media); load_model.assert_called_once()
        self.assertNotIn('error', result['rows']['1'])
        clip, sr = model.align.call_args.kwargs['audio'][0]
        self.assertEqual(sr, 16000)
        np.testing.assert_array_equal(clip, audio[int((1.-.32)*16000):int((4.+.32)*16000)])
        self.assertEqual(model.align.call_args.kwargs['text'], [self.source[0].text])
        self.assertEqual(result['peak_cuda_allocated_bytes'], 100)

    def test_launcher_uses_short_lived_module_and_timeout(self):
        def invoke(command,**kwargs):
            self.assertEqual(command[1:4],['-m','src.source_realign','--worker'])
            self.assertEqual(kwargs['timeout'],7)
            spec=read_json(Path(command[4]));self.assertEqual(spec['version'],VERSION)
            write_json(Path(command[5]),{'request_key':'key','rows':{}})
            return subprocess.CompletedProcess(command,0,'','')
        with patch('src.source_realign.subprocess.run',side_effect=invoke):
            result=_launch_worker({'version':VERSION},7)
        self.assertEqual(result,{'request_key':'key','rows':{}})


if __name__=='__main__':unittest.main()
