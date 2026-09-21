"""Integration checks for utterance evidence, paired display output and resume."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from main import build_parser
from src import quality, workflow
from src.config import TranslateConfig
from src.evidence import CueEvidence
from src.release import deliverables, inspect_deliverables
from src.translate import SrtBlock, make_snapshot, parse_srt, write_translated_srt
from src.workflow_state import StageState, artifact_path, file_hash, fingerprint, write_json


class WorkflowDisplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        previous_cwd = Path.cwd()
        os.chdir(temporary.name)
        self.addCleanup(os.chdir, previous_cwd)
        self.source = [SrtBlock(1, '00:00:01,000 --> 00:00:11,000',
                                'これは前半です。これは後半です。')]
        self.target = [SrtBlock(1, self.source[0].ts_line, '这是前半内容。这是后半内容。')]
        self.evidence = [CueEvidence(1, self.source[0].text, 'A',
                                     self.source[0].text, self.source[0].text)]
        self.decisions = [dict(line=1, status='accepted', text=self.source[0].text,
                               candidate='W', reason='Audio matches')]
        self.ledger = [dict(line=1, timestamp=self.source[0].ts_line, status='accepted',
                            reason='Fixture validation', source_decision=self.decisions[0])]
        self.job = workflow.Job(Path('output/work/final/episode.wav'),
                                Path('output/work/final/episode.srt'), 'work',
                                TranslateConfig(), utterance_mode=True)
        self.job.media.parent.mkdir(parents=True)
        self.job.media.write_bytes(b'fixture media')
        self.tokens = [[dict(text='これは前半です。', start=1, end=4),
                        dict(text='これは後半です。', start=4, end=11)]]
        self.metadata = artifact_path(self.job.source, '.asr.json')
        write_translated_srt(self.source, self.job.asr_source)
        write_json(self.metadata, dict(utterance_tokens=self.tokens))
        self.parts = [dict(source='これは前半です。', target='这是前半内容。'),
                      dict(source='これは後半です。', target='这是后半内容。')]
        self.args = SimpleNamespace(force=False, proofread=False, review=True)
        self.enterContext(patch.object(workflow, 'build_evidence', return_value=self.evidence))
        self.enterContext(patch.object(workflow, 'resolve_source', return_value=(self.source, self.decisions)))
        self.enterContext(patch.object(workflow, 'translate_scenes', return_value=self.target))
        self.qa = self.enterContext(patch.object(workflow, 'selective_qa',
                                                  return_value=(self.target, self.ledger)))
        self.enterContext(patch('src.display.plan_parts', return_value=self.parts))

    @property
    def translated(self):
        return self.job.config.resolve_translate_output(self.job.source)

    @property
    def semantic_target(self):
        return artifact_path(self.job.source, '.utterances.zh.srt')

    def run_language(self):
        workflow.process_language(self.job, self.args)

    def test_recipe_receives_raw_metadata_and_keeps_unchecked_qa(self):
        from dataclasses import asdict, replace
        recipe = Path('recipe.json')
        write_json(recipe, dict(version=1, model='local-test', stages=['edit-evidence']))
        self.job.config.coherence_recipe = recipe
        changed = [replace(self.target[0], text='这是前半内容。这是后半内容。')]
        self.ledger[0]['status'] = 'unchecked'
        self.ledger[0]['evidence'] = asdict(self.evidence[0])
        edits = [dict(line=1, before='原始草稿', after=changed[0].text)]
        with patch('src.coherence_workflow.refine_with_recipe', return_value=(changed, edits)) as refine:
            self.run_language()
        self.assertEqual(refine.call_args.args[2]['utterance_tokens'], self.tokens)
        self.assertEqual(self.ledger[0]['status'], 'unchecked')
        self.assertFalse(self.job.successful)
        self.assertTrue(artifact_path(self.job.source, '.coherence.json').exists())

    def test_failed_source_span_stage_keeps_workflow_incomplete(self):
        from src.entities import EntityLexicon
        from src.source_repair import SourceRepairResult
        self.job.config.source_span_repair = True
        self.args.warden_admin = 'http://127.0.0.1:8089'
        self.args.no_warden_unload = False
        result = SourceRepairResult(self.source, self.decisions, {'utterance_tokens': self.tokens},
                                    {'complete': False, 'error': 'Alignment unavailable'}, False)
        original = self.metadata.read_bytes()
        with patch('src.entities.extract_entity_lexicon', return_value=EntityLexicon('', fingerprint(''))), \
             patch('src.source_repair.repair_source_spans', return_value=result) as repair:
            self.run_language()
            self.assertFalse(self.job.successful)
            self.run_language()
        self.assertEqual(repair.call_count, 2)
        self.assertFalse(self.job.successful)
        self.assertEqual(self.metadata.read_bytes(), original)
        self.assertEqual(StageState(self.job.state_path).data['source_spans']['status'], 'failed')

    def test_source_span_alignment_sidecar_drives_display_and_resume(self):
        from copy import deepcopy
        from src.entities import EntityLexicon
        from src.source_repair import SourceRepairResult
        self.job.config.source_span_repair = True
        self.args.warden_admin = 'http://127.0.0.1:8089'
        self.args.no_warden_unload = False
        tokens = deepcopy(self.tokens)
        tokens[0][0]['end'] = 6
        tokens[0][1]['start'] = 6
        result = SourceRepairResult(self.source, self.decisions, {'utterance_tokens': tokens},
                                    {'complete': True}, True)
        original = self.metadata.read_bytes()
        with patch('src.entities.extract_entity_lexicon', return_value=EntityLexicon('', fingerprint(''))), \
             patch('src.source_repair.repair_source_spans', return_value=result) as repair:
            self.run_language()
            self.run_language()
            self.assertEqual(repair.call_count, 1)
            # Original exclusion documents can change without a changed summary.
            context = Path('original-context.txt')
            context.write_text('Updated original name definition', encoding='utf-8')
            self.job.config.context_files = [context]
            self.run_language()
            self.assertEqual(repair.call_count, 2)
            self.job.media.write_bytes(b'changed media fixture')
            self.run_language()
        self.assertEqual(repair.call_count, 3)
        self.assertTrue(self.job.successful)
        self.assertEqual(self.metadata.read_bytes(), original)
        self.assertTrue(parse_srt(self.translated)[0].ts_line.endswith('00:00:06,000'))
        self.assertTrue(artifact_path(self.job.source, '.source-resolution.json').exists())
        self.assertTrue(artifact_path(self.job.source, '.source-aligned.json').exists())

    def test_failed_entity_classification_cannot_appear_accepted(self):
        from src.entities import EntityLexicon, EntityClassifications, EntityOccurrence, EntityDecision
        self.job.config.entity_placeholders = True
        from dataclasses import asdict
        self.ledger[0]['evidence'] = asdict(self.evidence[0])
        lexicon = EntityLexicon('', fingerprint(''))
        occurrence = EntityOccurrence('E0001', 1, 0, 3, self.source[0].text[:3],
                                      fingerprint(self.source[0].text), ('g1',))
        labels = EntityClassifications({1: fingerprint(self.source[0].text)}, lexicon.key,
                                       [occurrence], [EntityDecision('E0001', 'UNKNOWN', 'invalid JSON', False)])
        def draft(*args, **kwargs):
            kwargs['entity_render']['1'] = 'unmarked'
            return self.target
        with patch('src.entities.extract_entity_lexicon', return_value=lexicon), \
             patch('src.entities.classify_entity_occurrences', return_value=labels), \
             patch.object(workflow, 'translate_scenes', side_effect=draft):
            self.run_language()
        self.assertFalse(self.job.successful)
        from src.workflow_state import read_json
        ledger = read_json(artifact_path(self.job.source, '.zh.quality.json'))
        self.assertEqual(ledger['cues'][0]['status'], 'unchecked')
        self.assertIn('classification incomplete', ledger['cues'][0]['reason'])

    def test_edited_paired_source_is_preserved_without_force(self):
        self.run_language()
        edited = self.job.source.read_text().replace('これは前半です。', '人工修正された前半。')
        self.job.source.write_text(edited)
        with self.assertRaisesRegex(RuntimeError, 'edited after QA'):
            self.run_language()
        self.assertEqual(self.job.source.read_text(), edited)

    def test_edited_internal_translation_is_also_preserved(self):
        self.run_language()
        edited = self.semantic_target.read_text().replace('这是前半内容。', '人工修订的前半内容。')
        self.semantic_target.write_text(edited)
        with self.assertRaisesRegex(RuntimeError, 'edited after QA'):
            self.run_language()
        self.assertEqual(self.semantic_target.read_text(), edited)

    def test_word_timing_change_regenerates_paired_display(self):
        self.run_language()
        self.assertTrue(parse_srt(self.translated)[0].ts_line.endswith('00:00:04,000'))
        self.tokens[0][0]['end'] = 6
        self.tokens[0][1]['start'] = 6
        write_json(self.metadata, dict(utterance_tokens=self.tokens))
        self.run_language()
        source, target = parse_srt(self.job.source), parse_srt(self.translated)
        self.assertTrue(target[0].ts_line.endswith('00:00:06,000'))
        self.assertEqual([row.ts_line for row in source], [row.ts_line for row in target])
        self.assertEqual(''.join(row.text for row in target), self.target[0].text)

    def test_display_implementation_change_invalidates_resume(self):
        self.run_language()
        self.run_language()
        self.assertEqual(self.qa.call_count, 1)
        original_hash = workflow.file_hash

        def changed_display(path):
            return 'updated display implementation' if path.name == 'display.py' else original_hash(path)

        with patch.object(workflow, 'file_hash', side_effect=changed_display):
            self.run_language()
        self.assertEqual(self.qa.call_count, 2)

    def test_missing_semantic_translation_is_reconstructed(self):
        self.run_language()
        self.semantic_target.unlink()
        self.run_language()
        self.assertTrue(self.semantic_target.exists())
        self.assertEqual(parse_srt(self.semantic_target)[0].text, self.target[0].text)
        self.assertTrue(self.job.successful)
        state = StageState(self.job.state_path)
        self.assertIn(self.semantic_target.name, state.data['quality']['outputs'])

    def test_organized_rerun_keeps_snapshots_in_review_and_release_files_clean(self):
        self.run_language()
        original = self.translated.read_bytes()
        self.args.force = True
        self.run_language()
        review = self.job.source.parent.parent / 'review'
        snapshot = review / 'episode.zh.pre-review.srt'
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertTrue((review / 'episode.pre-review.srt').exists())
        self.assertEqual(list(self.job.source.parent.glob('*pre-review.srt')), [])
        unit = self.job.source.parent.parent
        self.assertEqual(inspect_deliverables(unit, deliverables(unit)), [])

    def test_explicit_snapshot_directory_keeps_first_snapshot(self):
        source = Path('edited.zh.srt')
        source.write_text('Original text')
        directory = Path('review/snapshots')
        snapshot = make_snapshot(source, 'pre-review', directory=directory)
        source.write_text('Later edit')
        self.assertEqual(make_snapshot(source, 'pre-review', directory=directory), snapshot)
        self.assertEqual(snapshot.read_text(), 'Original text')
        self.assertFalse(Path('edited.zh.pre-review.srt').exists())

    def test_missing_raw_utterances_resume_through_ensemble_worker(self):
        args = build_parser().parse_args(['pipeline', '--evidence-first', '--asr-strategy',
                                         'ensemble', '--arbitrate', '--review', '--no-auto-context',
                                         '--organize', '-l', 'ja'])

        def worker(module, spec, directory, name):
            self.assertEqual(module, 'src.ensemble_asr')
            for job in spec['jobs']:
                source = Path(job['source'])
                self.assertEqual(source.name, 'episode.utterances.srt')
                write_translated_srt(self.source, source)
                metadata = Path(job['metadata'])
                write_json(metadata, dict(utterance_tokens=self.tokens))
                adjudication = Path(job['adjudication'])
                write_json(adjudication, [])
                state = StageState(Path(job['state']))
                state.save('asr', job['key'], [source, metadata])
                key = fingerprint([quality.VERSION, job['media_hash'], file_hash(source),
                                   spec['source_lang'], spec['batch_size']])
                state.save('arbitration', key, [adjudication])
            return 0

        with patch.object(workflow, 'run_worker', side_effect=worker) as calls:
            self.assertEqual(workflow.run_workflow(args), 0)
            self.assertEqual(calls.call_count, 1)
            self.job.asr_source.unlink()
            self.assertEqual(workflow.run_workflow(args), 0)
            self.assertEqual(calls.call_count, 2)
        self.assertTrue(self.job.asr_source.exists())
        self.assertTrue(self.translated.exists())
        self.assertEqual(self.job.asr_source.parent.name, 'review')


if __name__ == '__main__':
    unittest.main()
