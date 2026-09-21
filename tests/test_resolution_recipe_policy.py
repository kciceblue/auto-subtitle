"""Recipe-policy regressions against production modules; model/backend calls are mocked.
"""
from dataclasses import replace
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

from src import coherence_resolution as resolution
from src import coherence_workflow as workflow
from tests import test_coherence_workflow as existing


class RecipePolicyChecks(existing.CoherenceRecipeTests):
    def test_resolution_default_is_explicitly_conservative(self):
        self.data['stages']=['resolution-edit'];self.save()
        self.assertEqual(workflow.load_recipe(self.recipe)['resolution_diagnosis_policy'],'conservative')

    def test_nonresolution_recipe_does_not_inherit_policy(self):
        self.assertNotIn('resolution_diagnosis_policy',workflow.load_recipe(self.recipe))
        for value in ('conservative','recall-precheck',None):
            with self.subTest(policy=value):
                self.data['resolution_diagnosis_policy']=value;self.save()
                with self.assertRaisesRegex(ValueError,'requires a resolution-edit'):
                    workflow.load_recipe(self.recipe)

    def test_invalid_policy_rejected_before_context_or_backend(self):
        for value in (None,False,1,{},[], '', 'recall', 'Conservative'):
            with self.subTest(policy=value):
                self.data.update(stages=['resolution-edit'],resolution_diagnosis_policy=value);self.save()
                with patch.object(workflow,'_context') as context,patch.object(workflow,'temporary_local_writer') as backend, \
                     patch.object(resolution,'refine_resolution') as refine:
                    with self.assertRaisesRegex(ValueError,'Invalid resolution diagnosis policy'):
                        workflow.refine_with_recipe(self.source,self.draft,{},self.cfg,self.root/'invalid.json')
                    context.assert_not_called();backend.assert_not_called();refine.assert_not_called()

    def test_default_and_recall_policies_forward_to_staged_api(self):
        for policy in (None,'conservative','recall-precheck'):
            with self.subTest(policy=policy):
                self.data={'version':1,'model':'test-local','stages':['resolution-edit']}
                if policy is not None:self.data['resolution_diagnosis_policy']=policy
                self.save()
                with patch.object(resolution,'refine_resolution',return_value=(self.draft,[{'status':'UNVERIFIED'}])) as refine:
                    final,ledger=workflow.refine_with_recipe(self.source,self.draft,{},self.cfg,self.root/'forward.json')
                self.assertEqual(refine.call_args.kwargs['diagnosis_policy'],policy or 'conservative')
                self.assertFalse(ledger[0]['source_uncertainty_cleared']);self.assertEqual(final,self.draft)

    def test_policy_only_reaches_resolution_in_mixed_recipe(self):
        self.data.update(stages=['edit-evidence','resolution-edit'],resolution_diagnosis_policy='recall-precheck');self.save()
        with patch('src.coherence_refine.refine_coherence',return_value=(self.draft,[{'status':'UNVERIFIED'}])) as regular, \
             patch.object(resolution,'refine_resolution',return_value=(self.draft,[{'status':'UNVERIFIED'}])) as resolved:
            workflow.refine_with_recipe(self.source,self.draft,{},self.cfg,self.root/'mixed.json')
        self.assertNotIn('diagnosis_policy',regular.call_args.kwargs)
        self.assertEqual(resolved.call_args.kwargs['diagnosis_policy'],'recall-precheck')

    def test_cache_paths_and_bindings_separate_policy(self):
        self.data.update(stages=['resolution-edit']);self.save()
        bindings=[];paths=[]
        for policy in ('conservative','recall-precheck'):
            self.data['resolution_diagnosis_policy']=policy;self.save()
            bindings.append(workflow.recipe_binding(self.recipe,self.cfg,{}))
            with patch.object(resolution,'refine_resolution',return_value=(self.draft,[{'status':'UNVERIFIED'}])) as refine:
                workflow.refine_with_recipe(self.source,self.draft,{},self.cfg,self.root/'shared-cache.json')
                paths.append(refine.call_args.args[3])
        self.assertNotEqual(bindings[0],bindings[1]);self.assertNotEqual(paths[0],paths[1])
        self.assertEqual(bindings[0]['recipe']['resolution_diagnosis_policy'],'conservative')
        self.assertEqual(bindings[1]['recipe']['resolution_diagnosis_policy'],'recall-precheck')

    def test_policy_mutation_breaks_existing_preparation_binding(self):
        self.data.update(stages=['resolution-edit']);self.save()
        bound=workflow.recipe_binding(self.recipe,self.cfg,{})
        self.data['resolution_diagnosis_policy']='recall-precheck';self.save()
        with patch.object(workflow,'temporary_local_writer') as backend,patch.object(resolution,'refine_resolution') as refine:
            with self.assertRaisesRegex(ValueError,'changed'):
                workflow.refine_with_recipe(self.source,self.draft,{},self.cfg,self.root/'stale.json',expected_binding=bound)
            backend.assert_not_called();refine.assert_not_called()

    def test_staged_api_default_matches_recipe(self):
        param=inspect.signature(resolution.refine_resolution).parameters['diagnosis_policy']
        self.assertEqual(param.default,'conservative')
        self.assertEqual(param.kind,inspect.Parameter.KEYWORD_ONLY)


if __name__=='__main__':unittest.main(verbosity=2)
