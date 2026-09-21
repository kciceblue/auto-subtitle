"""Preserve compact reference facts and isolate cached context by source content."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import TranslateConfig
from src.work_context import MAX_VERBATIM_CHARS, prepare_work_context
from src.workflow_state import read_json, write_json


class WorkContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'context.json'
        self.config = TranslateConfig()

    def source(self, name: str, content: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def test_short_original_reference_bypasses_lossy_model_summary(self):
        content = '登場人物：マリは医師。妹の名前はユキ。\nユキ → 小雪（人名であり、地名ではない）。'
        path = self.source('reference.txt', content)
        with patch('src.work_context.call_llm', side_effect=AssertionError('No summary request')) as llm:
            context = prepare_work_context([path], ['作品名'], self.config, self.cache)
            cached = prepare_work_context([path], ['作品名'], self.config, self.cache)
        llm.assert_not_called()
        self.assertIn(content, context)
        self.assertIn('不能覆盖录音或补写台词', context)
        self.assertEqual(cached, context)
        self.assertEqual(read_json(self.cache)['mode'], 'verbatim')

    def test_distinct_corpora_cannot_reuse_a_cached_summary(self):
        path = self.source('reference.txt', '人物名はユキ。')
        with patch('src.work_context.call_llm') as llm:
            first = prepare_work_context([path], [], self.config, self.cache)
            first_key = read_json(self.cache)['key']
            path.write_text('人物名はマリ。', encoding='utf-8')
            second = prepare_work_context([path], [], self.config, self.cache)
        llm.assert_not_called()
        self.assertNotEqual(first_key, read_json(self.cache)['key'])
        self.assertNotEqual(first, second)
        self.assertIn('人物名はマリ。', second)
        self.assertNotIn('人物名はユキ。', second)

    def test_previous_summary_cache_is_replaced_with_original_material(self):
        path = self.source('reference.txt', 'ユキは人物名。')
        write_json(self.cache, {'key': 'work-context-1', 'context': 'ユキは地名。'})
        with patch('src.work_context.call_llm') as llm:
            context = prepare_work_context([path], [], self.config, self.cache)
        llm.assert_not_called()
        self.assertIn('ユキは人物名。', context)
        self.assertNotIn('ユキは地名。', context)

    def test_short_context_preserves_all_files_without_per_file_truncation(self):
        content = '人物に関する検証済み資料。' * 100
        primary = self.source('primary.txt', content)
        paths = [primary] + [self.source(f'empty-{i}.txt', '') for i in range(10)]
        with patch('src.work_context.call_llm') as llm:
            context = prepare_work_context(paths, [], self.config, self.cache)
        llm.assert_not_called()
        self.assertIn(content, context)

    def test_long_original_uses_compression_and_rejects_invented_glossary_evidence(self):
        path = self.source('long.txt', '医師の名前はデニス。' + '背景資料。' * MAX_VERBATIM_CHARS)
        response = json.dumps({'summary': '病院', 'terms': [
            {'source': 'デニス', 'translation': '丹尼斯', 'evidence': '医師の名前はデニス。'},
            {'source': '魔王', 'translation': '魔王', 'evidence': '魔王が登場'},
        ]}, ensure_ascii=False)
        with patch('src.work_context.call_llm', return_value=response) as llm:
            context = prepare_work_context([path], [], self.config, self.cache)
        llm.assert_called_once()
        self.assertLessEqual(len(llm.call_args.args[0]), 12000)
        self.assertIn('丹尼斯', context)
        self.assertNotIn('魔王', context)
        self.assertEqual(read_json(self.cache)['mode'], 'compressed')

    def test_edit_outside_long_excerpt_invalidates_source_provenance(self):
        path = self.source('long.txt', '背景資料。' * MAX_VERBATIM_CHARS)
        response = '{"summary":"背景","terms":[]}'
        with patch('src.work_context.call_llm', return_value=response) as llm:
            prepare_work_context([path], [], self.config, self.cache)
            original_key = read_json(self.cache)['key']
            path.write_text(path.read_text(encoding='utf-8') + '新資料。', encoding='utf-8')
            prepare_work_context([path], [], self.config, self.cache)
        self.assertEqual(llm.call_count, 2)
        self.assertNotEqual(read_json(self.cache)['key'], original_key)


if __name__ == '__main__':
    unittest.main()
