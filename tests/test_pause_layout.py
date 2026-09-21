"""Pause layout keeps exact paired text and rejects unsupported timing cuts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.config import TranslateConfig
from src.pause_layout import find_pause_cuts,plan_pause_parts

class PauseLayoutTests(unittest.TestCase):
    def setUp(self):
        self.source='今日は晴れ。明日は雨。'
        self.tokens=[dict(text='今日は晴れ。',start=0.,end=2.),dict(text='明日は雨。',start=5.,end=7.)]
        self.vad=[dict(start=0,end=2000),dict(start=5000,end=7000)]
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cache=Path(self.temp.name)/'pause.json'
    def test_exact_pause_cut_does_not_mutate_text_or_times(self):
        original=deepcopy((self.tokens,self.vad))
        cuts=find_pause_cuts(self.source,self.tokens,self.vad,1000)
        self.assertEqual(cuts,[dict(offset=6,gap_start=2.,gap_end=5.,gap_seconds=3.,vad_speech_fraction=0.)])
        self.assertEqual((self.tokens,self.vad),original)
    def test_inexact_tokens_or_speech_in_gap_do_not_split(self):
        self.assertEqual(find_pause_cuts('修正'+self.source,self.tokens,self.vad,1000),[])
        self.assertEqual(find_pause_cuts(self.source,self.tokens,[dict(start=0,end=7000)],1000),[])
        self.tokens[1]['end']=self.tokens[1]['start']
        self.assertEqual(find_pause_cuts(self.source,self.tokens,self.vad,1000),[])
    def test_no_tiny_internal_piece_from_multiple_gaps(self):
        tokens=[dict(text='明日は',start=0,end=1),dict(text='雨',start=4,end=5),dict(text='です。',start=8,end=9)]
        cuts=find_pause_cuts('明日は雨です。',tokens,[dict(start=0,end=1000)],1000)
        self.assertEqual([c['offset'] for c in cuts],[3])
    def test_valid_target_partition_cached_exactly(self):
        cuts=find_pause_cuts(self.source,self.tokens,self.vad,1000)
        with patch('src.pause_layout.call_llm',return_value=json.dumps({'targets':['今天晴天。','明天下雨。']})) as call:
            for _ in range(2):
                parts=plan_pause_parts(self.source,'今天晴天。明天下雨。',cuts,TranslateConfig(),self.cache)
        call.assert_called_once()
        self.assertEqual(''.join(p['source'] for p in parts),self.source)
        self.assertEqual(''.join(p['target'] for p in parts),'今天晴天。明天下雨。')
    def test_short_translation_uses_pause_layout_and_preserves_gap(self):
        from src.display import build_display
        from src.translate import SrtBlock
        source = [SrtBlock(1,'00:00:00,000 --> 00:00:07,000',self.source)]
        target = [SrtBlock(1,source[0].ts_line,'今天晴天。明天下雨。')]
        metadata = {'utterance_tokens':[self.tokens],'speech_regions':self.vad,'sample_rate':1000}
        with patch('src.pause_layout.call_llm',return_value=json.dumps({'targets':['今天晴天。','明天下雨。']})):
            raw,final,mapping = build_display(source,target,metadata,TranslateConfig(pause_layout=True),self.cache)
        self.assertEqual(len(final),2)
        self.assertEqual(''.join(r.text for r in raw),self.source)
        self.assertEqual(''.join(r.text for r in final),target[0].text)
        self.assertLess(mapping[0]['end'],mapping[1]['start'])
        self.assertEqual(mapping[0]['pause_layout']['status'],'partitioned')

    def test_rewording_punctuation_change_or_empty_part_fails_without_cache(self):
        cuts=find_pause_cuts(self.source,self.tokens,self.vad,1000)
        for values in (['今天天气晴朗。','明天下雨。'],['今天晴天','明天下雨。'],['今天晴天。明天下雨。','']):
            with patch('src.pause_layout.call_llm',return_value=json.dumps({'targets':values})):
                self.assertIsNone(plan_pause_parts(self.source,'今天晴天。明天下雨。',cuts,TranslateConfig(),self.cache))
        self.assertFalse(self.cache.exists())
if __name__=='__main__':unittest.main()
