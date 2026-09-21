"""Sentence integrity and display correspondence regressions."""
import tempfile
from copy import deepcopy
import unittest
from unittest.mock import patch
from pathlib import Path
from src.asr_consensus import compact, restore_sentence_boundaries, utterance_groups
from src.config import TranslateConfig
from src.display import build_display,validate_parts,plan_parts,restore_exact_parts,project_spelling_tokens
from src.translate import SrtBlock
class DisplayTests(unittest.TestCase):
    def test_punctuation_transfer_cannot_change_asr_words(self):
        text='私だって平気じゃあ一緒にやだ愛と一緒じゃ意味ない'
        result=restore_sentence_boundaries(text,'私だって平気。じゃあ一緒にやる。アイと一緒で意味ない。')
        self.assertEqual(compact(result),compact(text))
        self.assertIn('。',result)
    def test_kana_boundary_does_not_swallow_the_next_clause(self):
        text='強烈な未練だったわあれを親に見られた日には二度死ぬようなものね'
        template='強烈な未練だったわ。あれを親に見られた日には二度死ぬようなものね。'
        self.assertEqual(restore_sentence_boundaries(text,template),template)
    def test_pause_and_word_limit_do_not_split_negation(self):
        words=[dict(text='こっち',start=1,end=1.3),dict(text='来ん',start=2,end=2.5),dict(text='な。',start=4,end=4.1)]
        self.assertEqual([''.join(t['text'] for t in g) for g in utterance_groups(words)],['こっち来んな。'])
    def test_layout_cannot_drop_negation_or_add_a_fact(self):
        self.assertFalse(validate_parts([dict(source='来るな。',target='过来。')],'来るな。','别过来。'))
        self.assertTrue(validate_parts([dict(source='来るな。',target='别过来。')],'来るな。','别过来。'))
    def test_layout_punctuation_edits_are_discarded_but_word_edits_rejected(self):
        parts=[dict(source='ありがとう。',target='谢谢。'),dict(source='帰ろう。',target='回去吧。')]
        restored=restore_exact_parts(parts,'ありがとう帰ろう','谢谢，回去吧。')
        self.assertEqual(restored,[dict(source='ありがとう',target='谢谢，'),dict(source='帰ろう',target='回去吧。')])
        self.assertIsNone(restore_exact_parts(parts,'ありがとう帰ろう','谢谢，别回去。'))
    def test_layout_guard_allows_json_overhead_for_short_text(self):
        with tempfile.TemporaryDirectory() as d, patch('src.display.call_llm') as llm:
            llm.return_value = '{"parts":[{"source":"来るな。","target":"别过来。"}]}'
            result=plan_parts('来るな。','别过来。',TranslateConfig(),Path(d)/'cache.json')
            self.assertEqual(result,[dict(source='来るな。',target='别过来。')])
            self.assertGreaterEqual(llm.call_args.args[2].response_guard_floor,512)
    def test_clauses_sharing_one_alignment_are_joined_without_text_loss(self):
        text='これはとても長い文章です。続きもあります。'
        source=[SrtBlock(1,'00:00:01,000 --> 00:00:11,000',text)]
        target=[SrtBlock(1,source[0].ts_line,'这是一个很长的句子。还有后续内容。')]
        parts=[dict(source='これはとても長い文章です。',target='这是一个很长的句子。'),dict(source='続きもあります。',target='还有后续内容。')]
        meta=dict(utterance_tokens=[[dict(text=text,start=1,end=11)]])
        with tempfile.TemporaryDirectory() as d, patch('src.display.plan_parts',return_value=parts):
            a,b,m=build_display(source,target,meta,TranslateConfig(),Path(d)/'cache.json')
        self.assertEqual(len(b),1)
        self.assertEqual(b[0].text,target[0].text)
        self.assertGreater(m[0]['end'],m[0]['start'])
    def test_clause_starts_at_its_own_speech_after_a_long_pause(self):
        parts = [dict(source='前の話。 ', target='这是前面的完整一句话。 '),
                 dict(source='次の話。', target='这里是后面的完整一句话。')]
        source = [SrtBlock(1, '00:00:01,000 --> 00:00:10,000', ''.join(p['source'] for p in parts))]
        target = [SrtBlock(1, source[0].ts_line, ''.join(p['target'] for p in parts))]
        metadata = dict(utterance_tokens=[[dict(text='前の話。 ', start=1, end=2),
                                           dict(text='次の話。', start=7, end=10)]])
        with tempfile.TemporaryDirectory() as d, patch('src.display.plan_parts', return_value=parts):
            raw, final, rows = build_display(source, target, metadata, TranslateConfig(), Path(d)/'cache.json')
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]['start'], 6.92)  # Only the existing 80 ms display lead.
        self.assertAlmostEqual(rows[1]['end'], 10)
        self.assertLess(rows[0]['end'], 3)
        self.assertGreater(rows[1]['start'] - rows[0]['end'], 3)
        self.assertEqual(''.join(b.text for b in raw), source[0].text)
        self.assertEqual(''.join(b.text for b in final), target[0].text)
        self.assertTrue(all(row['alignment'] == 'word' for row in rows))

    def test_partition_inside_a_token_merges_only_the_shared_span(self):
        parts = [dict(source='連続', target='连续'),
                 dict(source='した言葉。', target='的话语在这里结束。'),
                 dict(source='続く説明。', target='接下来继续说明另一件事情。')]
        source = [SrtBlock(1, '00:00:01,000 --> 00:00:11,000', ''.join(p['source'] for p in parts))]
        target = [SrtBlock(1, source[0].ts_line, ''.join(p['target'] for p in parts))]
        metadata = dict(utterance_tokens=[[dict(text='連続した', start=1, end=3),
                                           dict(text='言葉。', start=3, end=4),
                                           dict(text='続く説明。', start=8, end=11)]])
        with tempfile.TemporaryDirectory() as d, patch('src.display.plan_parts', return_value=parts):
            raw, final, rows = build_display(source, target, metadata, TranslateConfig(), Path(d)/'cache.json')
        self.assertEqual(len(rows), 2)
        self.assertEqual(raw[0].text, '連続した言葉。')
        self.assertEqual(final[0].text, parts[0]['target'] + parts[1]['target'])
        self.assertAlmostEqual(rows[0]['end'], 4)
        self.assertAlmostEqual(rows[1]['start'], 7.92)
        self.assertLessEqual(rows[0]['end'], rows[1]['start'])
        self.assertEqual(''.join(b.text for b in raw), source[0].text)
        self.assertEqual(''.join(b.text for b in final), target[0].text)

    def test_changed_source_keeps_explicit_proportional_fallback(self):
        parts = [dict(source='新しい話。', target='这是前面的完整一句话。'),
                 dict(source='次の話。', target='这里是后面的完整一句话。')]
        source = [SrtBlock(1, '00:00:01,000 --> 00:00:10,000', ''.join(p['source'] for p in parts))]
        target = [SrtBlock(1, source[0].ts_line, ''.join(p['target'] for p in parts))]
        metadata = dict(utterance_tokens=[[dict(text='前の話。', start=1, end=2),
                                           dict(text='次の話。', start=7, end=10)]])
        with tempfile.TemporaryDirectory() as d, patch('src.display.plan_parts', return_value=parts):
            raw, final, rows = build_display(source, target, metadata, TranslateConfig(), Path(d)/'cache.json')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['alignment'] == 'utterance' for row in rows))
        self.assertAlmostEqual(rows[1]['start'], 6)
        self.assertEqual(''.join(b.text for b in raw), source[0].text)
        self.assertEqual(''.join(b.text for b in final), target[0].text)

    def test_applied_homophone_spelling_preserves_original_pause_and_tokens(self):
        parts = [dict(source='花です。 ', target='这里说的是花的完整一句话。 '),
                 dict(source='次の話。', target='这里是后面的完整一句话。')]
        source = [SrtBlock(1, '00:00:01,000 --> 00:00:10,000', ''.join(p['source'] for p in parts))]
        target = [SrtBlock(1, source[0].ts_line, ''.join(p['target'] for p in parts))]
        metadata = dict(utterance_tokens=[[dict(text='  はなです。 ', start=1, end=2),
                                           dict(text='次の話。  ', start=7, end=10)]])
        original = deepcopy(metadata)
        applied = {1: [dict(start=0, end=2, original='はな', replacement='花')]}
        with tempfile.TemporaryDirectory() as d, patch('src.display.plan_parts', return_value=parts):
            raw, final, rows = build_display(source, target, metadata, TranslateConfig(),
                                             Path(d)/'cache.json', spelling_patches=applied)
        self.assertEqual(metadata, original)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]['start'], 6.92)
        self.assertAlmostEqual(rows[1]['end'], 10)
        self.assertGreater(rows[1]['start'] - rows[0]['end'], 3)
        self.assertEqual(''.join(b.text for b in raw), source[0].text)
        self.assertEqual(''.join(b.text for b in final), target[0].text)
        for row in rows:
            self.assertEqual(row['alignment'], 'word')
            self.assertEqual(row['token_projection']['status'], 'spelling_projection')
            self.assertFalse(row['token_projection']['source_validated'])

    def test_multiple_spelling_spans_use_original_offsets_and_copy_times(self):
        tokens = [dict(text=' はなとこえ。 ', start=1.25, end=4.75, speaker='original')]
        original = deepcopy(tokens)
        projected, result = project_spelling_tokens(tokens, '花と声。', [
            dict(start=0, end=2, original='はな', replacement='花'),
            dict(start=3, end=5, original='こえ', replacement='声')])
        self.assertEqual(projected, [dict(text=' 花と声。 ', start=1.25, end=4.75, speaker='original')])
        self.assertEqual(tokens, original)
        self.assertIsNot(projected[0], tokens[0])
        self.assertEqual(result['patch_count'], 2)

    def test_homophone_source_without_applied_ledger_still_falls_back(self):
        tokens = [dict(text='はなです。', start=1, end=2)]
        projected, result = project_spelling_tokens(tokens, '花です。')
        self.assertIsNone(projected)
        self.assertEqual(result['reason'], 'source_changed_without_applied_spelling_patches')

    def test_cross_token_homophone_patch_has_explicit_display_fallback(self):
        source = [SrtBlock(1, '00:00:01,000 --> 00:00:03,000', '花です。')]
        target = [SrtBlock(1, source[0].ts_line, '这是花。')]
        metadata = dict(utterance_tokens=[[dict(text='は', start=1, end=1.5),
                                           dict(text='なです。', start=2, end=3)]])
        applied = {1: [dict(start=0, end=2, original='はな', replacement='花')]}
        with tempfile.TemporaryDirectory() as d:
            raw, final, rows = build_display(source, target, metadata, TranslateConfig(),
                                             Path(d)/'cache.json', spelling_patches=applied)
        self.assertEqual(rows[0]['alignment'], 'utterance')
        self.assertEqual(rows[0]['token_projection']['reason'], 'spelling_patch_crosses_token_boundary')
        self.assertEqual(raw[0].text, source[0].text)
        self.assertEqual(final[0].text, target[0].text)

    def test_spelling_ledger_rejects_stale_uncovered_or_overlapping_edits(self):
        tokens = [dict(text='はなとこえ。', start=1, end=4)]
        first = dict(start=0, end=2, original='はな', replacement='花')
        cases = [
            ('花とこえ。', [dict(first, original='鼻')], 'spelling_patch_original_mismatch'),
            ('花と声。', [first], 'spelling_patches_do_not_cover_selected_source'),
            ('花とこえ。', [first, first], 'overlapping_spelling_patches'),
            ('花とこえ。', [dict(first, start=False)], 'invalid_spelling_patch')]
        for source, patches, reason in cases:
            with self.subTest(reason=reason):
                projected, result = project_spelling_tokens(tokens, source, patches)
                self.assertIsNone(projected)
                self.assertEqual(result['reason'], reason)

    def test_spelling_projection_rejects_local_or_contextual_reading_change(self):
        cases = [
            ('はなです。', '声です。', dict(start=0, end=2, original='はな', replacement='声'),
             'spelling_patch_changes_reading'),
            ('生きる', 'なまきる', dict(start=0, end=1, original='生', replacement='なま'),
             'spelling_patches_change_contextual_reading')]
        for original, source, patch_row, reason in cases:
            with self.subTest(reason=reason):
                projected, result = project_spelling_tokens(
                    [dict(text=original, start=1, end=4)], source, [patch_row])
                self.assertIsNone(projected)
                self.assertEqual(result['reason'], reason)

    def test_display_hold_uses_free_space_and_preserves_source_target(self):
        source=[SrtBlock(1,'00:00:01,000 --> 00:00:01,080','来るな。'),SrtBlock(2,'00:00:03,000 --> 00:00:04,000','ありがとう。')]
        target=[SrtBlock(1,source[0].ts_line,'别过来。'),SrtBlock(2,source[1].ts_line,'谢谢。')]
        with tempfile.TemporaryDirectory() as d:
            a,b,m=build_display(source,target,{},TranslateConfig(),Path(d)/'cache.json')
        self.assertEqual(a[0].ts_line,b[0].ts_line)
        self.assertGreaterEqual(m[0]['end']-m[0]['start'],1-1e-9)
        self.assertLessEqual(m[0]['end'],m[1]['start'])
        self.assertEqual(b[0].text,'别过来。')


class NonlexicalLayoutTests(unittest.TestCase):
    def test_source_predicate_and_target_punctuation_stay_with_previous_text(self):
        from src.display import merge_nonlexical_parts
        parts = [{'source': '彼は私のことを', 'target': '他一直在想着我'},
                 {'source': '考えていた。', 'target': '。'}]
        merged = merge_nonlexical_parts(parts)
        self.assertEqual(merged, [{'source': '彼は私のことを考えていた。', 'target': '他一直在想着我。'}])
        self.assertEqual(len(parts), 2)

    def test_initial_punctuation_merges_forward_and_text_is_conserved(self):
        from src.display import merge_nonlexical_parts
        parts = [{'source': '「', 'target': '“'}, {'source': '帰る」と言った。', 'target': '回去”，他说。'},
                 {'source': 'またね。', 'target': '再见。'}]
        merged = merge_nonlexical_parts(parts)
        self.assertEqual(len(merged), 2)
        for field in ('source', 'target'):
            self.assertEqual(''.join(p[field] for p in parts), ''.join(p[field] for p in merged))

if __name__=='__main__':unittest.main()
