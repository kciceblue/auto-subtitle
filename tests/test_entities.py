"""Local entity extraction, source guards and marker conservation; no inference."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import TranslateConfig
from src.entities import (EntityAlias, EntityClassifications, EntityDecision,
    EntityDefinition, EntityLexicon, EntityMarkerError, EntityOccurrence, EntityPlan,
    _boundaries, _competing_name, _grounded, _occurrences, _tokenizer,
    build_entity_plan, classify_entity_occurrences, extract_entity_lexicon,
    restore_entity_source, restore_entity_target)
from src.translate import SrtBlock
from src.workflow_state import fingerprint, read_json, write_json

CONTEXT = 'ハルは人物。ハルの中文名は「春」、简称「小春」。ハルの条件付き別名は「はる」。\nハルカは人物。ハルカの中文名は「春香」。'

def definition_response():
    return {'entities': [{'source': 'ハル', 'target': '春', 'short_target': '小春',
        'source_quote': 'ハルは人物。', 'target_quote': 'ハルの中文名は「春」、简称「小春」。',
        'short_quote': 'ハルの中文名は「春」、简称「小春」。',
        'aliases': [{'text': 'はる', 'quote': 'ハルの条件付き別名は「はる」。', 'kind': 'conditional'}]}]}

def lexicon():
    return EntityLexicon(CONTEXT, fingerprint(CONTEXT), [
        EntityDefinition('g1', 'ハル', '春', '小春', 'ハルは人物。',
            'ハルの中文名は「春」、简称「小春」。', 'ハルの中文名は「春」、简称「小春」。',
            (EntityAlias('はる', 'ハルの条件付き別名は「はる」。', 'conditional'),)),
        EntityDefinition('g2', 'ハルカ', '春香', '春香', 'ハルカは人物。',
            'ハルカの中文名は「春香」。', 'ハルカの中文名は「春香」。')])

def source_rows(*texts):
    return [SrtBlock(i, '00:00:00,000 --> 00:00:09,000', text) for i, text in enumerate(texts, 1)]

def metadata(source, alternates=None):
    data = {'windows': [], 'raw_transcripts': [], 'used_transcripts': [],
            'source_selection': [], 'utterance_tokens': []}
    for i, block in enumerate(source):
        base = i*100; cursor = 0; group = []
        for token in _tokenizer().tokenize(block.text):
            end = cursor+len(token.surface)
            group.append({'text': token.surface, 'begin': cursor, 'finish': end,
                          'start': base+cursor*.1, 'end': base+end*.1})
            cursor = end
        data['windows'].append({'index': i, 'start': base, 'end': base+90,
                                 'core_start': base, 'core_end': base+90})
        data['source_selection'].append({'window': i, 'selected': 'q'})
        raw = {'q': block.text, 'w': block.text, 'n': block.text}
        raw.update((alternates or {}).get(i, {}))
        data['raw_transcripts'].append(raw)
        data['used_transcripts'].append(dict(raw))
        data['utterance_tokens'].append(group)
    return data

def classifications(source, lex, decision='g1'):
    occ = _occurrences(source, lex)
    return EntityClassifications({r.index: fingerprint(r.text) for r in source}, lex.key, occ,
        [EntityDecision(o.id, decision, 'Chosen source context') for o in occ])


class EntityExtractionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name)/'entities.json'; self.cfg = TranslateConfig()

    def test_exact_supplied_names_and_conditional_alias_are_cached(self):
        response = json.dumps(definition_response(), ensure_ascii=False)
        with patch('src.entities.call_llm', return_value=response) as llm:
            first = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
            second = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        llm.assert_called_once(); self.assertFalse(first.degraded)
        self.assertEqual(first.key, second.key)
        self.assertEqual(first.entities[0].short_target, '小春')
        self.assertEqual(first.entities[0].aliases[0].kind, 'conditional')
        self.assertEqual(read_json(self.cache)['response'], response)
        cfg = llm.call_args.args[2]
        self.assertFalse(cfg.extra_payload['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(cfg.stage, 'entity-lexicon')

    def test_entity_schema_requests_respect_caller_total_token_cap(self):
        from src.entities import _schema_config
        for maximum, stage_limit, expected in ((2048,4096,2048),(16384,4096,4096),(512,2048,512)):
            cfg = _schema_config(replace(self.cfg,max_tokens=maximum), 'entity-lexicon', {}, stage_limit)
            self.assertEqual(cfg.max_tokens,expected)
            self.assertEqual(cfg.extra_payload['max_tokens'],expected)

    def test_invented_target_and_alias_rejected_without_success_cache(self):
        invalid = definition_response(); invalid['entities'][0]['target'] = 'Unprovided Name'
        with patch('src.entities.call_llm', return_value=json.dumps(invalid)):
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertFalse(result.entities); self.assertTrue(result.degraded); self.assertFalse(self.cache.exists())
        invalid = definition_response(); invalid['entities'][0]['aliases'][0]['text'] = 'Absent'
        with patch('src.entities.call_llm', return_value=json.dumps(invalid)):
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertFalse(result.entities[0].aliases); self.assertTrue(result.degraded)

    def test_negative_original_quote_is_not_alias_authority(self):
        context = CONTEXT+'\nハルの別名はnot Yami。'
        self.assertFalse(_grounded(context, 'Yami', 'ハルの別名はnot Yami。', 'ハル'))
        self.assertFalse(_grounded('ハルではなく、別名は誤写「ヤミ」。', 'ヤミ', 'ハルではなく、別名は誤写「ヤミ」。', 'ハル'))
        self.assertTrue(_grounded('ハルの別名は「はる」。不是别人。', 'はる', 'ハルの別名は「はる」。不是别人。', 'ハル'))

    def test_context_edit_invalidates_full_context_key(self):
        with patch('src.entities.call_llm', return_value=json.dumps(definition_response())) as llm:
            first = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
            second = extract_entity_lexicon(CONTEXT+'\n追加背景。', self.cfg, self.cache)
        self.assertEqual(llm.call_count, 2); self.assertNotEqual(first.key, second.key)

    def test_duplicate_canonical_definitions_are_ambiguous(self):
        value = definition_response(); value['entities'].append(deepcopy(value['entities'][0]))
        with patch('src.entities.call_llm', return_value=json.dumps(value)):
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertFalse(result.entities); self.assertTrue(result.degraded)

    def test_original_paragraph_id_prevents_model_quote_copy_failures(self):
        response = {'entities': [{'source': 'ハル', 'target': '春', 'short_target': '小春',
                    'paragraph_id': 1, 'aliases': [{'text': 'はる', 'kind': 'conditional'}]}]}
        with patch('src.entities.call_llm', return_value=json.dumps(response)) as llm:
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertFalse(result.degraded)
        self.assertEqual(result.entities[0].aliases[0].quote, CONTEXT.split('\n')[0])
        self.assertEqual(json.loads(llm.call_args.args[0])['original_paragraphs']['1'], CONTEXT.split('\n')[0])
        response['entities'][0]['paragraph_id'] = 2
        with patch('src.entities.call_llm', return_value=json.dumps(response)):
            self.cache.unlink()
            wrong = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertTrue(wrong.degraded)
        self.assertFalse(wrong.entities)

    def test_single_shared_original_quote_is_expanded_without_invented_text(self):
        paragraph = CONTEXT.split('\n')[0]
        raw = json.dumps({'entities': [{'source': 'ハル', 'target': '春', 'short_target': '小春',
                         'quote': paragraph, 'aliases': [{'text': 'はる', 'kind': 'conditional'}]}]})
        with patch('src.entities.call_llm', return_value=raw):
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertFalse(result.degraded)
        self.assertEqual(result.entities[0].target_quote, paragraph)
        self.assertEqual(result.entities[0].aliases[0].quote, paragraph)

    def test_incomplete_quote_retry_preserves_rejected_raw_answer(self):
        invalid = definition_response()
        invalid['entities'][0]['target_quote'] = '「春」'
        raw = json.dumps(invalid)
        with patch('src.entities.call_llm', side_effect=[raw, json.dumps(definition_response())]) as llm:
            result = extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertEqual(llm.call_count, 2)
        self.assertFalse(result.degraded)
        rejected = read_json(self.cache.with_suffix('.rejections.json'))
        self.assertEqual(rejected[0]['raw_answer'], raw)
        self.assertIn('grounding checks', llm.call_args.args[1])

    def test_empty_context_bypasses_model_and_failure_is_retryable(self):
        with patch('src.entities.call_llm', side_effect=RuntimeError('offline')) as llm:
            self.assertFalse(extract_entity_lexicon('', self.cfg, self.cache).entities)
            extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
            extract_entity_lexicon(CONTEXT, self.cfg, self.cache)
        self.assertEqual(llm.call_count, 2); self.assertFalse(self.cache.exists())


class EntityClassificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name)/'classify.json'
        self.lex = lexicon(); self.source = source_rows('今日はハルが来ました。', '明日はハルが帰ります。')

    def test_compact_exact_id_classification_uses_only_source_and_declarations(self):
        response = json.dumps({'1': {'decision': 'g1', 'reason': 'Local name'},
                               '2': {'decision': 'ORDINARY_WORD', 'reason': 'Uncertain use'}})
        with patch('src.entities.call_llm', return_value=response) as llm:
            result = classify_entity_occurrences(self.source, self.lex, TranslateConfig(), self.cache)
            cached = classify_entity_occurrences(self.source, self.lex, TranslateConfig(), self.cache)
        llm.assert_called_once(); self.assertEqual(result.to_dict(), cached.to_dict())
        self.assertEqual(result.decisions[1].decision, 'ORDINARY_WORD')
        body = json.loads(llm.call_args.args[0]); self.assertEqual(len(body['source_neighborhood']), 2)
        self.assertNotIn('translation', body); self.assertEqual(llm.call_args.args[2].stage, 'entity-classify')

    def test_duplicate_or_missing_json_ids_never_default_to_entity(self):
        response = '{"1":{"decision":"g1","reason":"a"},"1":{"decision":"g1","reason":"b"}}'
        with patch('src.entities.call_llm', return_value=response):
            result = classify_entity_occurrences(self.source, self.lex, TranslateConfig(), self.cache)
        self.assertTrue(all(not r.checked and r.decision == 'UNKNOWN' for r in result.decisions))
        self.assertFalse(self.cache.exists())

    def test_changed_neighbor_invalidates_classification_cache(self):
        response = json.dumps({'1': {'decision': 'g1', 'reason': 'Local name'}})
        source = source_rows('今日はハルが来ました。', '雨です。')
        with patch('src.entities.call_llm', return_value=response) as llm:
            classify_entity_occurrences(source, self.lex, TranslateConfig(), self.cache)
            classify_entity_occurrences([source[0], replace(source[1], text='晴れです。')], self.lex, TranslateConfig(), self.cache)
        self.assertEqual(llm.call_count, 2)

    def test_longest_complete_alias_prevents_suffix_fragment_tasks(self):
        occ = _occurrences(source_rows('今日はハルカが来ました。'), self.lex)
        self.assertEqual([r.surface for r in occ], ['ハルカ'])


class EntityPlanTests(unittest.TestCase):
    def setUp(self):
        self.lex = lexicon(); self.source = source_rows('今日はハルが来ました。')
        self.meta = metadata(self.source); self.labels = classifications(self.source, self.lex)

    def make(self):
        return build_entity_plan(self.source, self.labels, self.lex, self.meta)

    def test_lossless_plan_preserves_original_source_and_timing(self):
        before = deepcopy((self.source, self.meta)); plan = self.make()
        row = plan.rows[1]; self.assertEqual(len(row.replacements), 1)
        self.assertEqual(restore_entity_source(row), self.source[0].text)
        self.assertEqual((self.source, self.meta), before)
        self.assertEqual(plan.marked_source[0].ts_line, self.source[0].ts_line)
        self.assertIn('identity remains unresolved', plan.notes[1][0])
        self.assertIn(row.replacements[0].marker, plan.translation_instruction([1]))
        self.assertEqual(plan.translation_instruction([]), '')

    def test_repaired_utterance_offsets_are_not_raw_window_identity_evidence(self):
        for token in self.meta['utterance_tokens'][0]:
            token['offset_scope'] = 'utterance'
        plan = self.make()
        self.assertFalse(plan.rows[1].replacements)
        self.assertEqual(plan.ledger[0]['reason'], 'No unique exact aligned-token ownership')

    def test_serialization_roundtrip_and_stale_spans_reject(self):
        plan = self.make(); restored = EntityPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(plan.key, restored.key)
        data = plan.to_dict(); data['rows']['1']['replacements'][0]['start'] += 1
        with self.assertRaises(ValueError): EntityPlan.from_dict(data)

    def test_restore_before_length_checks_and_reject_bad_marker_counts(self):
        row = self.make().rows[1]; marker = row.replacements[0].marker
        self.assertEqual(restore_entity_target(marker+'来了。', row), '春来了。')
        for target in ['春来了。', marker+marker, marker+'⟦E9999⟧', '⟦broken⟧']:
            with self.assertRaises(EntityMarkerError): restore_entity_target(target, row)

    def test_ordinary_unknown_and_unchecked_are_unforced(self):
        for decision in ['ORDINARY_WORD', 'UNKNOWN']:
            labels = classifications(self.source, self.lex, decision)
            plan = build_entity_plan(self.source, labels, self.lex, self.meta)
            self.assertFalse(plan.rows[1].replacements); self.assertEqual(plan.marked_source, self.source)
        labels = classifications(self.source, self.lex); labels.decisions[0] = replace(labels.decisions[0], checked=False)
        self.assertFalse(build_entity_plan(self.source, labels, self.lex, self.meta).rows[1].replacements)

    def test_raw_longer_different_identity_blocks_clipped_name(self):
        alt = '今日はハルカが来ました。'
        data = metadata(self.source, {0: {'w': alt, 'n': alt}})
        plan = build_entity_plan(self.source, self.labels, self.lex, data)
        self.assertFalse(plan.rows[1].replacements)
        self.assertIn('different/longer', plan.ledger[0]['reason'])

    def test_other_name_elsewhere_is_not_local_competing_evidence(self):
        text = '今日はハルが来ました。その後でハルカが帰ります。'
        source = source_rows(text); labels = classifications(source, self.lex)
        labels.decisions = [replace(r, decision='g1' if labels.occurrences[i].surface=='ハル' else 'g2') for i, r in enumerate(labels.decisions)]
        plan = build_entity_plan(source, labels, self.lex, metadata(source))
        self.assertEqual(len(plan.rows[1].replacements), 2)

    def test_edited_source_or_incomplete_token_mapping_stays_unforced(self):
        changed = [replace(self.source[0], text=self.source[0].text+'追加。')]
        self.assertFalse(build_entity_plan(changed, self.labels, self.lex, self.meta).rows[1].replacements)
        bad = deepcopy(self.meta); bad['utterance_tokens'][0][0]['text'] = 'changed'
        self.assertFalse(build_entity_plan(self.source, self.labels, self.lex, bad).rows[1].replacements)
        self.assertFalse(build_entity_plan(self.source, self.labels, self.lex, {}).rows[1].replacements)

    def test_duplicate_labels_cannot_create_multiple_placeholders(self):
        self.labels.decisions.append(self.labels.decisions[0])
        plan = self.make(); self.assertFalse(plan.rows[1].replacements)
        self.assertIn('duplicate', plan.ledger[0]['reason'])

    def test_near_reading_or_inferred_alias_cannot_become_marker(self):
        source = source_rows('今日はハロが来ました。'); block = source[0]
        occ = EntityOccurrence('E0001', 1, 3, 5, 'ハロ', fingerprint(block.text), ('g1',))
        labels = EntityClassifications({1: fingerprint(block.text)}, self.lex.key, [occ], [EntityDecision(occ.id, 'g1', 'near')])
        plan = build_entity_plan(source, labels, self.lex, metadata(source))
        self.assertFalse(plan.rows[1].replacements)


if __name__ == '__main__': unittest.main()
