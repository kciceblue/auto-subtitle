"""Entity translation integration: lossless markers, retry scope and cache binding."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser
from src.config import TranslateConfig
from src.entities import EntityPlan, EntityReplacement, EntityRowPlan
from src.evidence import CueEvidence
from src.quality import TRANSLATE, parse_records, query, translate_scenes
from src.translate import SrtBlock
from src.workflow import configuration_key
from src.workflow_state import fingerprint


class EntityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / 'requests.json'
        self.source = [SrtBlock(i, f'00:00:0{i},000 --> 00:00:0{i+1},000', text)
                       for i, text in enumerate(('ユキ。', '元気です。'), 1)]
        self.evidence = [CueEvidence(b.index, b.text, 'A', b.text, b.text) for b in self.source]
        self.cfg = TranslateConfig(chunk_size=24, pack_scenes=True)
        self.plan = self.make_plan('小雪')

    def make_plan(self, target):
        replacement = EntityReplacement('E0001', 0, 2, 'ユキ', '⟦E0001⟧', 'g1', target,
                                         f'ユキ → {target}')
        first = EntityRowPlan(1, self.source[0].ts_line, self.source[0].text,
                              '⟦E0001⟧。', (replacement,))
        second = EntityRowPlan(2, self.source[1].ts_line, self.source[1].text,
                               self.source[1].text, ())
        return EntityPlan([replace(self.source[0], text=first.marked_source), self.source[1]],
                          {1: first, 2: second}, [], {}, 'context', target,
                          {b.index: fingerprint(b.text) for b in self.source}, 'metadata')

    def test_restore_before_length_gate_and_never_leak_markers(self):
        with patch('src.quality.call_llm', return_value='[1] ⟦E0001⟧。\n[2] 我很好。') as call:
            result = translate_scenes(self.source, self.evidence, self.cfg, self.cache,
                                      entity_plan=self.plan)
        self.assertEqual([r.text for r in result], ['小雪。', '我很好。'])
        self.assertIn('⟦E0001⟧', call.call_args.args[0])
        self.assertIn('exact marker', call.call_args.args[1])
        self.assertEqual(self.source[0].text, 'ユキ。')

    def test_duplicate_marker_retries_only_failed_then_plain_fallback(self):
        responses = ['[1] ⟦E0001⟧和⟦E0001⟧。\n[2] 我很好。',
                     '[1] 小雪。', '[1] 小雪。']
        receipt = {}
        with patch('src.quality.call_llm', side_effect=responses) as call:
            result = translate_scenes(self.source, self.evidence, self.cfg, self.cache,
                                      entity_plan=self.plan, entity_render=receipt)
        self.assertEqual(call.call_count, 3)
        self.assertIn('本次仅处理这些ID: 1。', call.call_args_list[1].args[0])
        self.assertNotIn('⟦E0001⟧', call.call_args_list[2].args[0])
        self.assertEqual(result[1].text, '我很好。')
        self.assertEqual(receipt['1'], 'fallback_identity_unchecked')
        self.assertEqual(receipt['2'], 'unmarked')

    def test_restored_cache_not_reparsed_as_marker_output_and_target_change_invalidates(self):
        with patch('src.quality.call_llm', return_value='[1] ⟦E0001⟧。\n[2] 我很好。') as call:
            for _ in range(2):
                rows = query([1,2], self.source, self.evidence, self.cfg, TRANSLATE,
                             'translation', self.cache, entity_plan=self.plan)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(rows[1], ['小雪。'])
            changed = self.make_plan('雪')
            rows = query([1,2], self.source, self.evidence, self.cfg, TRANSLATE,
                         'translation', self.cache, entity_plan=changed)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(rows[1], ['雪。'])

    def test_stale_source_plan_fails_before_model_request(self):
        changed = [replace(self.source[0], text='ユキです。'), self.source[1]]
        with patch('src.quality.call_llm') as call:
            with self.assertRaisesRegex(ValueError, 'identical chosen'):
                translate_scenes(changed, self.evidence, self.cfg, self.cache, entity_plan=self.plan)
        call.assert_not_called()

    def test_direct_translation_and_unmarked_rows_reject_foreign_marker(self):
        self.assertEqual(parse_records('[1] ⟦E0009⟧。', [1], 'translation', self.evidence), {})
        with patch('src.quality.call_llm', side_effect=['[1] ⟦E0001⟧。\n[2] ⟦E0001⟧。', '[1] 我很好。']):
            rows = query([1,2], self.source, self.evidence, self.cfg, TRANSLATE,
                         'translation', self.cache, entity_plan=self.plan)
        self.assertEqual(rows[2], ['我很好。'])

    def test_targeted_repair_cannot_reintroduce_reserved_markers(self):
        self.assertEqual(parse_records("[1] FIX|⟦E1⟧です。", [1], "patch", self.evidence), {})
        self.assertEqual(parse_records("[1] FIX|小雪。", [1], "patch", self.evidence), {1: ["FIX", "小雪。"]})

    def test_cli_flag_and_config_key(self):
        args = build_parser().parse_args(['pipeline', '--entity-placeholders'])
        self.assertTrue(args.entity_placeholders)
        self.assertNotEqual(configuration_key(self.cfg),
                            configuration_key(replace(self.cfg, entity_placeholders=True)))


if __name__ == '__main__':
    unittest.main()
