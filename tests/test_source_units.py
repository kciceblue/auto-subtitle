"""Translation-unit cuts require real saved token and VAD support."""
from copy import deepcopy
import unittest
from src.source_units import pause_token_groups


class SourceUnitTests(unittest.TestCase):
    def setUp(self):
        self.tokens = [dict(text='今日は',start=0.,end=1.,begin=0,finish=3),
                       dict(text='晴れ',start=1.,end=2.,begin=3,finish=5),
                       dict(text='明日は雨',start=5.,end=7.,begin=5,finish=10)]
        self.vad = [dict(start=0,end=2000),dict(start=5000,end=7000)]

    def test_split_retains_exact_tokens_offsets_times_and_inputs(self):
        old = deepcopy((self.tokens,self.vad))
        groups,cuts = pause_token_groups(self.tokens,self.vad,1000)
        self.assertEqual(len(groups),2)
        self.assertEqual([t for g in groups for t in g],self.tokens)
        self.assertEqual(cuts[0]['offset'],5)
        groups[0][0]['text']='changed copy'
        self.assertEqual((self.tokens,self.vad),old)

    def test_speech_in_gap_and_missing_vad_keep_original(self):
        for vad in ([],[dict(start=0,end=7000)]):
            groups,cuts = pause_token_groups(self.tokens,vad,1000)
            self.assertEqual(groups,[self.tokens]);self.assertEqual(cuts,[])

    def test_internal_whitespace_is_never_dropped(self):
        self.tokens[1]['text'] += ' '
        groups,cuts = pause_token_groups(self.tokens,self.vad,1000)
        self.assertEqual(groups,[self.tokens]);self.assertEqual(cuts,[])

    def test_tiny_interval_and_zero_boundary_do_not_split(self):
        for start in (6.8,7.):
            self.tokens[2]['start']=start
            groups,cuts = pause_token_groups(self.tokens,self.vad,1000)
            self.assertEqual(groups,[self.tokens]);self.assertEqual(cuts,[])

    def test_multiple_gaps_keep_all_characters(self):
        self.tokens += [dict(text='週末は雪',start=10.,end=12.,begin=10,finish=15)]
        self.vad += [dict(start=10000,end=12000)]
        groups,cuts = pause_token_groups(self.tokens,self.vad,1000)
        self.assertEqual(len(groups),3)
        self.assertEqual([t for g in groups for t in g],self.tokens)


if __name__=='__main__':unittest.main()
