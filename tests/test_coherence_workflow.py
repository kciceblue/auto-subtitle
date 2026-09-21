"""Offline recipe boundary and pipeline cache tests."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser, cmd_pipeline
from src.coherence_workflow import load_recipe, recipe_binding, refine_with_recipe
from src.config import TranslateConfig
from src.translate import SrtBlock


class CoherenceRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = self.root / 'recipe.json'
        self.data = {'version': 1, 'model': 'test-local', 'stages': ['edit-evidence', 'critic-edit']}
        self.save()
        self.cfg = TranslateConfig(endpoint='http://127.0.0.1:8089/v1/chat/completions',
                                   coherence_recipe=self.recipe)
        self.source = [SrtBlock(1, '00:00:01,000 --> 00:00:02,000', 'はい。')]
        self.draft = [replace(self.source[0], text='是的。')]

    def save(self):
        self.recipe.write_text(json.dumps(self.data), encoding='utf-8')

    def test_unknown_feedback_fields_rejected(self):
        self.data['external_findings'] = ['must never reach writer']
        self.save()
        with self.assertRaises(ValueError):
            load_recipe(self.recipe)

    def test_duplicate_recipe_key_rejected(self):
        self.recipe.write_text('{"version":1,"version":1}')
        with self.assertRaises(ValueError):
            load_recipe(self.recipe)

    def test_external_writer_rejected(self):
        self.data['model'] = 'gpt-6-astra'
        self.save()
        with self.assertRaises(ValueError):
            load_recipe(self.recipe)

    def test_weights_relative_to_recipe_and_hash_pinned(self):
        model = self.root / 'writer.gguf'
        model.write_bytes(b'weights')
        binary = self.root / 'server'
        binary.write_text('#!/bin/sh\n')
        binary.chmod(0o700)
        self.data.update(weights='writer.gguf', server_binary='server',
                         sha256=hashlib.sha256(model.read_bytes()).hexdigest())
        self.save()
        self.assertEqual(load_recipe(self.recipe)['weights'], str(model))
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_recipe(self.recipe)

    def test_binding_changes_with_raw_evidence_and_original_context(self):
        original = recipe_binding(self.recipe, self.cfg, {'raw': 1})
        self.assertNotEqual(original, recipe_binding(self.recipe, self.cfg, {'raw': 2}))
        context = self.root / 'context.txt'
        context.write_text('Original background')
        changed = replace(self.cfg, context_files=[context])
        first = recipe_binding(self.recipe, changed, {'raw': 1})
        context.write_text('Updated background')
        self.assertNotEqual(first, recipe_binding(self.recipe, changed, {'raw': 1}))

    def test_recipe_mutation_rejected_before_any_writer(self):
        bound = recipe_binding(self.recipe, self.cfg, {})
        self.data['stages'] = ['scan-edit']
        self.save()
        with patch('src.coherence_refine.refine_coherence') as call:
            with self.assertRaisesRegex(ValueError, 'changed'):
                refine_with_recipe(self.source, self.draft, {}, self.cfg, self.root / 'cache.json',
                                   expected_binding=bound)
            call.assert_not_called()

    def test_stages_chain_only_local_outputs_and_stay_unverified(self):
        seen = []
        def run(source, draft, metadata, cfg, cache, **kw):
            seen.append((draft[0].text, cfg.extra_payload, cache, kw['mode']))
            output = [replace(draft[0], text=draft[0].text + '好。')]
            return output, [{'before': draft[0].text, 'after': output[0].text, 'status': 'UNVERIFIED'}]
        with patch('src.coherence_refine.refine_coherence', side_effect=run):
            final, ledger = refine_with_recipe(self.source, self.draft, {}, self.cfg, self.root / 'cache.json')
        self.assertEqual(seen[1][0], '是的。好。')
        self.assertNotEqual(seen[0][2], seen[1][2])
        self.assertEqual(final[0].text, '是的。好。好。')
        self.assertEqual(ledger[0]['status'], 'UNVERIFIED')
        self.assertFalse(ledger[0]['source_uncertainty_cleared'])
        self.assertEqual(self.draft[0].text, '是的。')

    def test_backend_cleanup_on_refinement_failure(self):
        self.data.update(weights='unused', server_binary='unused', sha256='0'*64)
        cleanup = []
        @contextmanager
        def backend(*args, **kwargs):
            try:
                yield 'http://127.0.0.1:18101/v1/chat/completions', {'model_alias': 'test-local'}
            finally:
                cleanup.append(True)
        with patch('src.coherence_workflow.recipe_binding', return_value={'recipe': {
                **self.data, 'max_tokens': 4096, 'sampler': {}, 'context_size': 16384,
                'admin_url': 'http://127.0.0.1:8089/admin', 'restore_model': 'original',
                'batch_size': 20, 'critic_budget': 1024, 'full_swa': False, 'edit_budget': 0, 'prior_score': None, 'cpu_moe_layers': 0, 'threads': 4}}), \
             patch('src.coherence_workflow.temporary_local_writer', side_effect=backend), \
             patch('src.coherence_refine.refine_coherence', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                refine_with_recipe(self.source, self.draft, {}, self.cfg, self.root / 'cache.json')
        self.assertEqual(cleanup, [True])
        self.assertFalse((self.root / 'cache.json').exists())

    def test_message_layout_is_bound_and_propagated(self):
        before = recipe_binding(self.recipe, self.cfg, {})
        self.data['separate_instruction'] = True
        self.save()
        self.assertNotEqual(before, recipe_binding(self.recipe, self.cfg, {}))
        observed = []
        def run(source, draft, metadata, cfg, cache, **kwargs):
            observed.append(cfg.separate_instruction)
            return draft, [{'status':'UNVERIFIED'} for _ in draft]
        with patch('src.coherence_refine.refine_coherence', side_effect=run):
            refine_with_recipe(self.source, self.draft, {}, self.cfg, self.root/'cache.json')
        self.assertEqual(observed, [True, True])
        self.data['separate_instruction'] = 'true'
        self.save()
        with self.assertRaises(ValueError):
            load_recipe(self.recipe)

    def test_scene_critic_recipe_is_admitted_and_routes_to_local_refinement(self):
        self.data['stages'] = ['scene-critic-edit']
        self.save()
        self.assertEqual(load_recipe(self.recipe)['stages'], ['scene-critic-edit'])
        edits = [{'line': 1, 'before': self.draft[0].text, 'after': self.draft[0].text,
                  'status': 'UNVERIFIED', 'source_uncertainty_cleared': False}]
        with patch('src.coherence_refine.refine_coherence', return_value=(self.draft, edits)) as refine:
            final, ledger = refine_with_recipe(self.source, self.draft, {}, self.cfg,
                                               self.root / 'scene-cache.json')
        self.assertEqual(refine.call_args.kwargs['mode'], 'scene-critic-edit')
        self.assertEqual(final, self.draft)
        self.assertEqual(ledger[0]['status'], 'UNVERIFIED')
        self.assertFalse(ledger[0]['source_uncertainty_cleared'])

    def test_resolution_stage_receives_local_draft_and_preserves_unresolved_ledger(self):
        self.data.update(stages=['edit-evidence', 'resolution-edit'], batch_size=30,
                         context_size=32768, critic_budget=1536, edit_budget=1024,
                         max_tokens=8192)
        self.save()
        intermediate = [replace(self.draft[0], text='好的。')]
        final = [replace(self.draft[0], text='好。')]
        edits = [{'status': 'UNVERIFIED', 'unresolved': True}]
        with patch('src.coherence_refine.refine_coherence', return_value=(intermediate, edits)), \
             patch('src.coherence_resolution.refine_resolution', return_value=(final, edits)) as resolution:
            actual, ledger = refine_with_recipe(self.source, self.draft, {}, self.cfg,
                                                self.root / 'resolution-cache.json')
        self.assertEqual(resolution.call_args.args[1], intermediate)
        self.assertEqual(resolution.call_args.kwargs['max_repair_retries'], 2)
        self.assertEqual(resolution.call_args.kwargs['context_size'], 32768)
        self.assertEqual(resolution.call_args.kwargs['edit_budget'], 1024)
        self.assertEqual(actual, final)
        self.assertTrue(ledger[0]['steps'][-1]['unresolved'])
        self.assertEqual(ledger[0]['status'], 'UNVERIFIED')
        self.assertFalse(ledger[0]['source_uncertainty_cleared'])

    def test_resolution_limits_reject_before_backend(self):
        for field, value in [('batch_size', 41), ('max_tokens', 16384),
                             ('critic_budget', 4097), ('edit_budget', 4097)]:
            with self.subTest(field=field):
                self.data = {'version': 1, 'model': 'test-local', 'stages': ['resolution-edit'],
                             'max_tokens': 8192, field: value}
                self.save()
                with patch('src.coherence_workflow.temporary_local_writer') as backend:
                    with self.assertRaisesRegex(ValueError, 'Resolution stage'):
                        refine_with_recipe(self.source, self.draft, {}, self.cfg,
                                           self.root / 'resolution-cache.json')
                    backend.assert_not_called()

    def test_resolution_timeout_rejected_before_backend(self):
        self.data['stages'] = ['resolution-edit']
        self.save()
        with patch('src.coherence_workflow.temporary_local_writer') as backend:
            with self.assertRaisesRegex(ValueError, 'timeout'):
                refine_with_recipe(self.source, self.draft, {}, replace(self.cfg, timeout=601),
                                   self.root / 'resolution-cache.json')
            backend.assert_not_called()

    def test_cli_requires_evidence_first(self):
        args = build_parser().parse_args(['pipeline', '--coherence-recipe', str(self.recipe)])
        self.assertEqual(cmd_pipeline(args), 1)


if __name__ == '__main__':
    unittest.main()
