"""Raw evidence and ownership regressions for bounded source span selection."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from src.source_spans import (build_span_candidates, parse_span_selections,
                              project_span_selections, selection_schema,
                              source_selection_body)


Q = '明日の午後はエイガを見に行きます。'
WN = '明日の午後は映画を見に行きます。'


def fixture(q=Q, w=WN, n=WN, *, window_id=0, utterance_id=17):
    raw = dict(q=q, w=w, n=n)
    return dict(windows=[dict(index=window_id, start=0., end=10., core_start=0., core_end=10.)],
        raw_transcripts=[raw], used_transcripts=[deepcopy(raw)],
        source_selection=[dict(window=window_id, selected='q')],
        segments=[dict(index=utterance_id, text=q, start=1., end=8.)],
        utterance_tokens=[[dict(text=q, begin=0, finish=len(q), start=1., end=8.)]])


def select(candidates, decision=None):
    windows = sorted({row['window'] for row in candidates})
    answer = {str(i): dict(decision=decision or next(row['id'] for row in candidates
              if row['window'] == window), reason='Both local alternatives support the change')
              for i, window in enumerate(windows, 1)}
    return parse_span_selections(json.dumps(answer), candidates)


class SourceSpanTests(unittest.TestCase):
    def test_exact_two_family_candidate_retains_full_raw_provenance(self):
        metadata = fixture()
        before = deepcopy(metadata)
        candidates = build_span_candidates(metadata)
        self.assertEqual(len(candidates), 1)
        row = candidates[0]
        self.assertEqual((row['before'], row['replacement']), ('エイガ', '映画'))
        self.assertEqual({r['family'] for r in row['support']}, {'w', 'n'})
        for support in row['support']:
            self.assertEqual(metadata['raw_transcripts'][0][support['family']]
                             [support['start']:support['end']], row['replacement'])
        self.assertEqual(row['source_status'], 'unresolved')
        self.assertFalse(row['applied_to_pipeline'])
        self.assertEqual(metadata, before)

    def test_all_pos_keeps_bounded_predicate_adjective_evidence(self):
        q = 'これから先の計画は簡単に決められます。'
        wn = 'これから先の計画は大胆に決められます。'
        rows = build_span_candidates(fixture(q, wn, wn))
        self.assertEqual([(r['before'], r['replacement']) for r in rows], [('簡単', '大胆')])
        self.assertGreater(rows[0]['support'][0]['reading_edit_distance'], 0)
        self.assertEqual(rows[0]['source_status'], 'unresolved')

    def test_one_family_or_other_input_channels_cannot_support_a_replacement(self):
        metadata = fixture(n=Q)
        metadata['raw_transcripts'][0].update(helper=WN, captions=WN, evaluator=WN)
        self.assertEqual(build_span_candidates(metadata), [])
        metadata['raw_transcripts'][0]['n'] = ''
        self.assertEqual(build_span_candidates(metadata), [])

    def test_insert_delete_and_changed_protected_surfaces_are_excluded(self):
        self.assertEqual(build_span_candidates(fixture(WN, WN.replace('映画', '映画館'),
                                                      WN.replace('映画', '映画館'))), [])
        self.assertEqual(build_span_candidates(fixture(), protected_surfaces=['エイガ']), [])
        # A lock can extend outside a candidate boundary: inspect the whole text.
        self.assertEqual(build_span_candidates(fixture(), protected_surfaces=['午後はエイガ']), [])

    def test_repeated_anchors_do_not_choose_an_arbitrary_occurrence(self):
        q = '明日の朝から午後までエイガを見に行く予定ですが今は仕事をしています。'
        wn = q.replace('エイガ', '映画')
        self.assertEqual(build_span_candidates(fixture(q+q, wn+wn, wn+wn)), [])

    def test_sparse_global_windows_have_exact_local_schema_and_no_extra_inputs(self):
        metadata = fixture(window_id=91)
        metadata['draft'] = 'PRIVATE TARGET SENTINEL'
        metadata['raw_transcripts'][0]['helper'] = 'HELPER SENTINEL'
        rows = build_span_candidates(metadata)
        schema = selection_schema(rows)
        self.assertEqual(schema['required'], ['1'])
        body = source_selection_body(metadata, rows)
        self.assertNotIn('PRIVATE TARGET SENTINEL', body)
        self.assertNotIn('HELPER SENTINEL', body)
        choice = select(rows)[0]
        self.assertEqual(choice['window'], 91)
        self.assertEqual(choice['source_status'], 'unresolved')

    def test_invalid_duplicate_missing_or_wrong_window_choices_never_pass(self):
        rows = build_span_candidates(fixture())
        valid = dict(decision=rows[0]['id'], reason='Raw alternatives agree')
        bad = ['{}', json.dumps({'91': valid}),
               '{"1":'+json.dumps(valid)+',"1":'+json.dumps(valid)+'}',
               '{"1":{"decision":"KEEP","decision":"UNSURE","reason":"x"}}',
               json.dumps({'1': dict(valid, decision='A999')}),
               json.dumps({'1': dict(valid, reason=' ')}),
               json.dumps({'1': dict(valid, reason='x'*181)}),
               json.dumps({'1': dict(valid, replacement=WN)})]
        for answer in bad:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                parse_span_selections(answer, rows)

    def test_unequal_length_projection_returns_offsets_without_changing_raw_times(self):
        metadata = fixture()
        original = deepcopy(metadata)
        rows = build_span_candidates(metadata)
        source = {17: Q}
        result = project_span_selections(source, metadata, select(rows), rows)
        self.assertEqual(result['source'], {17: WN})
        self.assertEqual(result['changed_ids'], [17])
        decision = result['decisions'][0]
        self.assertEqual(decision['projected_span'], dict(start=6, end=9, before='エイガ', replacement='映画'))
        self.assertEqual(decision['timing_status'], 'requires_realignment')
        self.assertEqual(decision['source_status'], 'unresolved')
        self.assertEqual(metadata, original)
        self.assertEqual(source, {17: Q})

    def test_span_may_cover_multiple_tokens_but_not_multiple_utterances(self):
        metadata = fixture()
        metadata['utterance_tokens'] = [[
            dict(text=Q[:7], begin=0, finish=7, start=1., end=3.),
            dict(text=Q[7:], begin=7, finish=len(Q), start=3., end=8.)]]
        rows = build_span_candidates(metadata)
        accepted = project_span_selections({17: Q}, metadata, select(rows), rows)
        self.assertEqual(accepted['changed_ids'], [17])
        metadata['segments'] = [dict(index=17, text=Q[:7]), dict(index=88, text=Q[7:])]
        metadata['utterance_tokens'] = [[token] for token in metadata['utterance_tokens'][0]]
        rejected = project_span_selections({17: Q[:7], 88: Q[7:]}, metadata, select(rows), rows)
        self.assertEqual(rejected['changed_ids'], [])
        self.assertIn('crosses utterance', rejected['decisions'][0]['rejection'])

    def test_overlapping_core_ownership_and_duplicate_raw_coverage_are_rejected(self):
        for duplicate_core in (True, False):
            with self.subTest(duplicate_core=duplicate_core):
                metadata = fixture()
                rows = build_span_candidates(metadata)
                if duplicate_core:
                    metadata['windows'].append(dict(metadata['windows'][0], index=8))
                    metadata['raw_transcripts'].append(deepcopy(metadata['raw_transcripts'][0]))
                    metadata['used_transcripts'].append(deepcopy(metadata['used_transcripts'][0]))
                    metadata['source_selection'].append(dict(window=8, selected='q'))
                else:
                    metadata['segments'].append(dict(metadata['segments'][0], index=88))
                    metadata['utterance_tokens'].append(deepcopy(metadata['utterance_tokens'][0]))
                result = project_span_selections({17: Q, 88: Q}, metadata, select(rows), rows)
                self.assertEqual(result['changed_ids'], [])
                self.assertEqual(result['decisions'][0]['status'], 'rejected')

    def test_stale_raw_support_source_and_non_q_backbone_cannot_apply(self):
        for change in ('raw', 'source', 'selected', 'used', 'candidate'):
            with self.subTest(change=change):
                metadata = fixture()
                rows = build_span_candidates(metadata)
                source = {17: Q}
                if change == 'raw':
                    metadata['raw_transcripts'][0]['w'] += '追加'
                elif change == 'source':
                    source[17] = Q + '追加'
                elif change == 'selected':
                    metadata['source_selection'][0]['selected'] = 'w'
                elif change == 'used':
                    metadata['used_transcripts'][0]['q'] = WN
                else:
                    rows[0]['support'][0]['start'] += 1
                result = project_span_selections(source, metadata, select(rows), rows)
                self.assertEqual(result['changed_ids'], [])
                self.assertEqual(result['source'], source)
                self.assertEqual(result['decisions'][0]['status'], 'rejected')

    def test_merged_window_projection_uses_token_owner_not_first_window(self):
        metadata = fixture(window_id=9)
        first = '少し待ってください。'
        first_raw = dict(q=first, w=first, n=first)
        metadata['windows'][0].update(start=10., end=20., core_start=10., core_end=20.)
        metadata['windows'].insert(0, dict(index=2, start=0., end=10., core_start=0., core_end=10.))
        metadata['raw_transcripts'].insert(0, first_raw)
        metadata['used_transcripts'].insert(0, deepcopy(first_raw))
        metadata['source_selection'].insert(0, dict(window=2, selected='q'))
        prefix = first.rstrip('。')
        merged = prefix + Q
        metadata['segments'][0]['text'] = merged
        metadata['utterance_tokens'] = [[
            dict(text=prefix, begin=0, finish=len(first), start=1., end=8.),
            dict(text=Q, begin=0, finish=len(Q), start=11., end=18.)]]
        before = deepcopy(metadata)
        rows = build_span_candidates(metadata)
        result = project_span_selections({17: merged}, metadata, select(rows), rows)
        self.assertEqual(result['source'], {17: prefix+WN})
        self.assertEqual(result['decisions'][0]['window'], 9)
        self.assertEqual(result['decisions'][0]['projected_span']['start'], len(prefix)+6)
        self.assertEqual(metadata, before)

    def test_disjoint_windows_in_one_utterance_apply_against_original_offsets(self):
        metadata = fixture(window_id=2)
        metadata['windows'].append(dict(index=9, start=10., end=20., core_start=10., core_end=20.))
        metadata['raw_transcripts'].append(deepcopy(metadata['raw_transcripts'][0]))
        metadata['used_transcripts'].append(deepcopy(metadata['used_transcripts'][0]))
        metadata['source_selection'].append(dict(window=9, selected='q'))
        merged = Q.rstrip('。') + Q
        metadata['segments'][0]['text'] = merged
        metadata['utterance_tokens'] = [[
            dict(text=Q.rstrip('。'), begin=0, finish=len(Q), start=1., end=8.),
            dict(text=Q, begin=0, finish=len(Q), start=11., end=18.)]]
        rows = build_span_candidates(metadata)
        result = project_span_selections({17: merged}, metadata, select(rows), rows)
        self.assertEqual(result['changed_ids'], [17])
        self.assertEqual(result['source'][17], WN.rstrip('。')+WN)
        self.assertEqual([r['projected_span']['start'] for r in result['decisions']],
                         [6, len(Q.rstrip('。'))+6])
        # A retry containing only the second window gets request-local ID 1.
        retry = [row for row in rows if row['window'] == 9]
        self.assertEqual(selection_schema(retry)['required'], ['1'])
        self.assertEqual(select(retry)[0]['window'], 9)
        wrong = {'1': dict(decision=retry[0]['id'], reason='Wrong window'),
                 '2': dict(decision='KEEP', reason='Keep')}
        with self.assertRaises(ValueError):
            parse_span_selections(json.dumps(wrong), rows)
        choices = select(rows)
        with self.assertRaises(ValueError):
            project_span_selections({17: merged}, metadata, choices+choices[:1], rows)

    def test_missing_raw_token_characters_block_projection(self):
        metadata = fixture()
        metadata['utterance_tokens'] = [[
            dict(text=Q[:7], begin=0, finish=7, start=1., end=3.),
            dict(text=Q[8:], begin=8, finish=len(Q), start=3., end=8.)]]
        metadata['segments'][0]['text'] = Q[:7]+Q[8:]
        rows = build_span_candidates(metadata)
        source = {17: metadata['segments'][0]['text']}
        result = project_span_selections(source, metadata, select(rows), rows)
        self.assertEqual(result['changed_ids'], [])
        self.assertIn('missing', result['decisions'][0]['rejection'])

    def test_keep_and_unsure_preserve_source_uncertainty(self):
        metadata = fixture()
        rows = build_span_candidates(metadata)
        for choice in ('KEEP', 'UNSURE'):
            result = project_span_selections({17: Q}, metadata, select(rows, choice), rows)
            self.assertEqual(result['source'], {17: Q})
            self.assertEqual(result['changed_ids'], [])
            self.assertEqual(result['decisions'][0]['source_status'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
