"""Offline regressions for independent ASR windows and phrase reconstruction."""
import tempfile
import unittest
from pathlib import Path

from src.asr_consensus import (AudioWindow, aligned_tokens, audio_windows, choose_transcript,
                               phrase_grade, project_phrase, subtitle_groups)
from src.hotwords import filter_asr_hotwords
from src.title_context import clean_release_title


class EnsembleTests(unittest.TestCase):
    def test_windows_cover_quiet_audio_and_have_overlap(self):
        speech=[{'start':16000,'end':32000},{'start':45*16000,'end':46*16000}]
        windows=audio_windows(78.54,speech)
        self.assertEqual(windows[0].core_start,0)
        self.assertEqual(windows[-1].core_end,78.54)
        for a,b in zip(windows,windows[1:]):
            self.assertEqual(a.core_end,b.core_start)
            self.assertGreater(a.end,b.start)
        self.assertTrue(all(w.end-w.start<=21.61 for w in windows))

    def test_missing_conversation_recovered_by_two_recognizers(self):
        candidates={'w':'作詞・作曲・編曲 初音ミク','q':'二人で一緒に作った。私の方は綺麗にできなくて。',
                    'n':'二人で一緒に作った私の方はきれいにできなくて'}
        selected,score=choose_transcript(candidates)
        self.assertEqual(selected,'q')
        self.assertGreater(score,.6)

    def test_two_short_transcripts_cannot_erase_a_long_shared_tail(self):
        selected,score=choose_transcript({'w':'キスはその激しか','n':'まあキスはその激しかった',
            'q':'一晩帰らなかったから。なんでもない平気。健全な女の子です。まあキスはその激しか。'})
        self.assertEqual(selected,'q')
        self.assertLess(score,.82)

    def test_aligner_rounding_at_clip_edge_does_not_replace_dialogue(self):
        w=AudioWindow(0,0,17.896,0,17.096)
        tokens=aligned_tokens('大丈夫。も。',[dict(text='大丈夫',start_time=4,end_time=5),
            dict(text='も',start_time=17.92,end_time=17.92)],w)
        self.assertEqual(''.join(t['text'] for t in tokens),'大丈夫。')

    def test_phrase_projection_does_not_include_neighbor_dialogue(self):
        full='二人で作った。愛がくれた。嬉しかった。'
        other='ふたりで作った。アイがくれた。うれしかった。'
        begin=full.index('愛');end=full.index('嬉')
        projected=project_phrase(full,other,begin,end)
        self.assertIn('アイ',projected)
        self.assertNotIn('ふたり',projected)
        self.assertNotIn('うれし',projected)

    def test_spelling_agreement_and_meaningful_homophones(self):
        self.assertEqual(phrase_grade('ありがとう。',{'q':'ありがとう。','w':'ありがとう','n':'ありがとう'},'q'),'A')
        self.assertEqual(phrase_grade('去勢手術',{'q':'去勢手術','w':'虚勢手術','n':'きょせいしゅじゅつ'},'q'),'E-音')

    def test_negation_and_numbers_cannot_be_fuzzy_votes(self):
        text='明日の朝はここには来ません'
        self.assertEqual(phrase_grade(text,{'q':text,'w':'明日の朝はここには来ます','n':'明日の朝はここには来ます'},'q'),'C')
        text='明日の朝10時にここに来てください'
        self.assertEqual(phrase_grade(text,{'q':text,'w':text.replace('10','11'),'n':text.replace('10','11')},'q'),'C')

    def test_alignment_owns_overlap_and_preserves_punctuation(self):
        window=AudioWindow(1,9.2,20.8,10,20)
        items=[dict(text='前',start_time=.1,end_time=.3),dict(text='本当',start_time=1,end_time=2),
               dict(text='です',start_time=2,end_time=3),dict(text='後',start_time=11,end_time=11.2)]
        tokens=aligned_tokens('前。本当です。後。',items,window)
        self.assertEqual(''.join(t['text'] for t in tokens),'本当です。')
        self.assertAlmostEqual(tokens[0]['start'],10.2)
        self.assertEqual(len(subtitle_groups(tokens)),1)

    def test_cues_split_at_pauses_and_duration(self):
        tokens=[dict(text='一',start=1,end=2),dict(text='二',start=4,end=5),dict(text='三',start=5,end=9)]
        self.assertEqual(len(subtitle_groups(tokens,max_seconds=4)),3)

    def test_alignment_matches_a_word_across_punctuation_without_changing_text(self):
        w=AudioWindow(0,0,3,0,3)
        tokens=aligned_tokens('ちゃ。んと',[dict(text='ちゃんと',start_time=1,end_time=2)],w)
        self.assertEqual(tokens,[dict(text='ちゃ。んと',start=1,end=2,begin=0,finish=5)])

    def test_bad_aligner_output_is_rejected(self):
        with self.assertRaises(ValueError):
            aligned_tokens('本当。',[dict(text='嘘',start_time=0,end_time=1)],AudioWindow(0,0,3,0,3))


    def test_alignment_failure_retries_without_fabricating_times(self):
        from dataclasses import make_dataclass
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.ensemble_asr import align_with_retry
        Item = make_dataclass('Item', ['text', 'start_time', 'end_time'])
        bad = dict(text='本当', start_time=2, end_time=1)
        model = Mock()
        model.align.side_effect = [[SimpleNamespace(items=[Item(**bad)])],
                                   [SimpleNamespace(items=[Item('本当', 1, 2)])]]
        chosen, tokens, failures = align_with_retry(model, [], 16000, AudioWindow(0,0,3,0,3),
                                                   {'q':'本当。','w':'本当'}, 'q', [bad])
        self.assertEqual(chosen, 'w')
        self.assertEqual(tokens[0]['start'], 1)
        self.assertEqual(tokens[0]['end'], 2)
        self.assertEqual(len(failures), 2)
        self.assertEqual(model.align.call_count, 2)

    @staticmethod
    def merge_fixture(rows):
        from src.asr import Segment
        segments, evidence, tokens = [], [], []
        for index, (window, start, end, text) in enumerate(rows, 1):
            segments.append(Segment(index, start, end, text))
            evidence.append(dict(line=index, window=window, w=text, whisper=text,
                                 n=text, q=text, grade='C', source_model='q', note='raw'))
            tokens.append([dict(text=text, start=start, end=end)])
        return segments, evidence, tokens

    def test_cross_window_merge_does_not_absorb_next_same_window_sentence(self):
        from src.ensemble_asr import merge_window_utterances
        fixture = [(0, 18, 19, '古い手紙'), (1, 20, 20.8, 'を見つけた。'),
                   (1, 21, 22, 'が、開けなかった。')]
        segments, evidence, tokens = merge_window_utterances(*self.merge_fixture(fixture))
        self.assertEqual([s.text for s in segments], ['古い手紙を見つけた。', 'が、開けなかった。'])
        self.assertEqual([(s.start, s.end) for s in segments], [(18, 20.8), (21, 22)])
        self.assertEqual([s.index for s in segments], [1, 2])
        self.assertEqual([row['line'] for row in evidence], [1, 2])
        self.assertEqual(evidence[0]['window'], 0)
        self.assertEqual(evidence[0]['last_window'], 1)
        self.assertEqual(evidence[0]['windows'], [0, 1])
        self.assertEqual(evidence[1]['windows'], [1])
        for segment, row, words in zip(segments, evidence, tokens):
            self.assertEqual(''.join(t['text'] for t in words), segment.text)
            self.assertEqual(row['w'], segment.text)
            self.assertEqual(row['q'], segment.text)
            self.assertEqual(row['grade'], 'C')
        self.assertEqual(''.join(s.text for s in segments), ''.join(row[3] for row in fixture))

    def test_continuation_across_three_windows_records_all_origins(self):
        from src.ensemble_asr import merge_window_utterances
        fixture = [(4, 98, 99, '明日は'), (5, 100, 119, '駅へ向かって'),
                   (6, 120, 121, '歩きます。')]
        segments, evidence, tokens = merge_window_utterances(*self.merge_fixture(fixture))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, '明日は駅へ向かって歩きます。')
        self.assertEqual((segments[0].start, segments[0].end), (98, 121))
        self.assertEqual(evidence[0]['window'], 4)
        self.assertEqual(evidence[0]['last_window'], 6)
        self.assertEqual(evidence[0]['windows'], [4, 5, 6])
        self.assertEqual([(t['start'], t['end']) for t in tokens[0]], [(98, 99), (100, 119), (120, 121)])

    def test_window_provenance_does_not_relax_existing_pause_or_sentence_gate(self):
        from src.ensemble_asr import merge_window_utterances
        fixtures = [
            [(0, 18, 19, '古い手紙'), (1, 22, 23, 'を見つけた。')],
            [(0, 18, 19, '話は終わり。'), (1, 20, 21, '次へ進もう。')]]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                segments, evidence, tokens = merge_window_utterances(*self.merge_fixture(fixture))
                self.assertEqual([s.text for s in segments], [row[3] for row in fixture])
                self.assertEqual([row['windows'] for row in evidence], [[0], [1]])
                self.assertEqual([row['last_window'] for row in evidence], [0, 1])

    def test_release_metadata_is_not_a_japanese_hotword(self):
        title='S01E13 - [GROUP&OTHER][Summer_Pockets][13][1080P][GB]'
        self.assertEqual(clean_release_title(title),'Summer Pockets')
        self.assertEqual(filter_asr_hotwords(['GROUP','Summer Pockets','1080P','稲荷'],'Japanese'),['稲荷'])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'vocab.txt';p.write_text('BDSM\n虚勢->去勢')
            self.assertEqual(filter_asr_hotwords(['BDSM','虚勢','去勢'],'Japanese',p),['BDSM','去勢'])


if __name__=='__main__':
    unittest.main()
