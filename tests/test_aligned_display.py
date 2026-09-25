import unittest

from src import aligned_display as ad


def tokens(spec):
    """spec: [(text, start, end)] -> contiguous tokens with character spans."""
    out, cursor = [], 0
    for text, start, end in spec:
        out.append(dict(text=text, start=start, end=end, begin=cursor, finish=cursor + len(text)))
        cursor += len(text)
    return out


class JapaneseUnitChecks(unittest.TestCase):
    def test_sentence_and_pause_cuts(self):
        text = 'そっか。でも俺たちは一緒だ'
        toks = tokens([('そっか。', 0, .5), ('でも', .6, .8), ('俺たちは', 2.0, 2.5), ('一緒だ', 2.6, 3.0)])
        units = ad.japanese_units(text, toks)
        self.assertEqual([u['text'] for u in units], ['そっか。', 'でも俺たちは一緒だ'])

    def test_short_sentence_is_not_cut_at_a_bare_pause(self):
        text = '君ともっと一緒にいたいのに'
        toks = tokens([('君ともっと', 0, 1), ('一緒に', 1.1, 1.6), ('いたいのに', 2.6, 3.2)])
        self.assertEqual([u['text'] for u in ad.japanese_units(text, toks)], [text])

    def test_long_unit_never_splits_before_a_particle(self):
        text = 'お友達を作ろうと思いましたそれで'
        toks = tokens([('お友達', 0, 3), ('を作ろうと', 4, 6), ('思いました', 6.1, 7.5), ('それで', 8.2, 9)])
        units = ad.japanese_units(text, toks)
        self.assertEqual([u['text'] for u in units], ['お友達を作ろうと思いました', 'それで'])

    def test_long_unit_splits_at_comma(self):
        text = 'ハロウィンとか、クリスマスとか楽しめる'
        toks = tokens([('ハロウィンとか、', 0, 4), ('クリスマスとか', 4.1, 7.5), ('楽しめる', 7.6, 9)])
        self.assertEqual([u['text'] for u in ad.japanese_units(text, toks)],
                         ['ハロウィンとか、', 'クリスマスとか楽しめる'])

    def test_trailing_unaligned_text_stays(self):
        text = 'ん…んくっ…'
        units = ad.japanese_units(text, tokens([('ん…', 0, .2), ('んくっ', .3, .5)]))
        self.assertEqual(''.join(u['text'] for u in units), text)

    def test_proportional_fallback_covers_text(self):
        units = ad.proportional_units('あいう。えお?か', 10, 16)
        self.assertEqual(''.join(u['text'] for u in units), 'あいう。えお?か')
        self.assertAlmostEqual(units[0]['start'], 10); self.assertAlmostEqual(units[-1]['end'], 16)


class ChinesePieceChecks(unittest.TestCase):
    def test_monotone_partition_is_exact_and_ordered(self):
        units = [{'text': 'あいうえお。'}, {'text': 'かき。'}, {'text': 'くけこさしす。'}]
        pieces = ad.monotone_partition(units, '甲乙丙丁戊。己庚。辛壬癸子丑寅。')
        self.assertEqual(''.join(p['target'] for p in pieces), '甲乙丙丁戊。己庚。辛壬癸子丑寅。')
        self.assertEqual([p['units'] for p in pieces], [[0, 0], [1, 1], [2, 2]])

    def test_monotone_partition_merges_uncovered_units(self):
        units = [{'text': 'あ。'}, {'text': 'い。'}, {'text': 'う。'}]
        pieces = ad.monotone_partition(units, '一整句。')
        self.assertEqual(pieces, [{'units': [0, 2], 'target': '一整句。'}])

    def test_parse_targets_requires_exact_concatenation(self):
        self.assertEqual(ad._parse_targets('{"targets":["你好，","世界。"]}', 2, '你好，世界。'), ['你好，', '世界。'])
        self.assertIsNone(ad._parse_targets('{"targets":["你好","世界。"]}', 2, '你好，世界。'))
        self.assertIsNone(ad._parse_targets('{"targets":[]}', 2, '你好，世界。'))

    def test_split_long_piece_at_punctuation(self):
        text = '不，干脆把一辈子的活动都给做了吧。这样很棒不是吗？万圣节和圣诞节什么的，可以一次性全部享受到。'
        parts = ad.split_piece(text, 0, 10)
        self.assertEqual(''.join(p['text'] for p in parts), text)
        self.assertTrue(all(len(ad.convention(p['text'])) <= ad.MAX_CUE_CHARS for p in parts))
        self.assertTrue(all(a['end'] == b['start'] for a, b in zip(parts, parts[1:])))

    def test_convention_and_preservation(self):
        self.assertEqual(ad.convention('静久，我想到一个主意。'), '静久 我想到一个主意')
        self.assertEqual(ad.convention('为什么？我、我也…'), '为什么？我、我也…')
        self.assertTrue(ad.preserved('你好，世界。', '你好 世界'))
        self.assertFalse(ad.preserved('你好，世界。', '你好 世'))

    def test_finalize_timing_no_overlap_minimum_duration(self):
        cues = ad.finalize_timing([{'start': 1.0, 'end': 1.1, 'text': '啊'}, {'start': 1.2, 'end': 3.0, 'text': '好'},
                                   {'start': 3.0, 'end': 3.0, 'text': '嗯'}])
        for a, b in zip(cues, cues[1:]):
            self.assertLess(a['end'], b['start'])
        self.assertTrue(all(c['end'] > c['start'] for c in cues))
        self.assertGreaterEqual(cues[-1]['end'] - cues[-1]['start'] + 1e-9, ad.MIN_CUE_SECONDS)


class RepairAndLayerChecks(unittest.TestCase):
    def test_collapsed_run_is_spread_over_the_gap(self):
        toks = tokens([('今日はとても暑いですね', 0, 2.0), ('海に行きたい', 2.4, 2.5), ('です', 2.6, 2.9)])
        fixed, count = ad.repair_timing(toks, 0, 5)
        self.assertEqual(count, 1)  # the 0.9 s it needs fits the 0.6 s gap only loosely: use the whole gap
        self.assertAlmostEqual(fixed[1]['start'], 2.0); self.assertAlmostEqual(fixed[1]['end'], 2.6)

    def test_collapsed_run_in_a_long_gap_stays_near_its_aligned_position(self):
        toks = tokens([('今日はとても暑いですね', 0, 2.0), ('海に行きたい', 2.4, 2.5), ('です', 6.0, 6.3)])
        fixed, _ = ad.repair_timing(toks, 0, 8)
        self.assertAlmostEqual(fixed[1]['start'], 2.0); self.assertAlmostEqual(fixed[1]['end'], 2.9)

    def test_mostly_implausible_owner_raises(self):
        toks = tokens([('あいうえおかきくけこ', 0, .1), ('さ', .2, .5)])
        with self.assertRaises(ValueError):
            ad.repair_timing(toks, 0, 5)

    def test_place_piece_does_not_bridge_long_silence(self):
        units = [{'start': 0.0, 'end': 1.0}, {'start': 9.0, 'end': 10.0}]
        parts = ad.place_piece('生日快乐。真是个美好的暑假啊。', units)
        self.assertEqual([p['text'] for p in parts], ['生日快乐。', '真是个美好的暑假啊。'])
        self.assertLessEqual(parts[0]['end'], 1.0); self.assertGreaterEqual(parts[1]['start'], 9.0)

    def test_sentences_keep_leading_ellipsis_with_next_utterance(self):
        self.assertEqual(ad._sentences('早上好，小明。…早上好。'), ['早上好，小明。', '…早上好。'])

    def test_convention_absorbs_punctuation_next_to_ellipsis(self):
        self.assertEqual(ad.convention('生日快乐。…'), '生日快乐…')
        self.assertEqual(ad.convention('但那里没有你…。…'), '但那里没有你……')

    def test_reading_minimum_applied_before_lead_out(self):
        cues = ad.finalize_timing([{'start': 0.0, 'end': .3, 'text': '那个人夸奖了我的毛绒玩具'},
                                   {'start': 2.0, 'end': 3.0, 'text': '嗯'}])
        self.assertGreaterEqual(cues[0]['end'] - cues[0]['start'] + 1e-9, 12 / ad.READING_CPS)
        self.assertLess(cues[0]['end'], cues[1]['start'])


if __name__ == '__main__':
    unittest.main()
