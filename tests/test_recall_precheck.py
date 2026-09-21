"""Synthetic recall/precheck contracts; every HTTP/model call is mocked."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_coherence_resolution as a
from src import coherence_resolution as R

# Run the13 established contracts against the new conservative default as well.
ConservativeRegressionChecks=a.ResolutionChecks
CapacityRegressionChecks=a.CapacityChecks

class RecallChecks(unittest.TestCase):
    setUp=a.ResolutionChecks.setUp
    invoke=a.ResolutionChecks.invoke
    saved=a.ResolutionChecks.saved

    def recall(self,answers,**kwargs):return self.invoke(answers,diagnosis_policy='recall-precheck',**kwargs)

    def test_conservative_request_body_keeps_original_fields(self):
        _,_,call=self.invoke([a.diagnosis('ISSUE','OK'),a.encoded({'1':'甲句。'}),a.resolution('resolved')])
        for item in call.call_args_list:
            self.assertEqual(set(json.loads(item.args[0])),{'cycle','current_draft_sha256','items'})

    def test_unsupported_suspicion_never_rewritten(self):
        final,ledger,call=self.recall([a.diagnosis('ISSUE','OK'),a.resolution('diagnosis_unsupported')])
        self.assertEqual(final,self.draft);self.assertEqual(call.call_count,2)
        self.assertEqual(ledger[0]['repair_attempts'],0)
        self.assertEqual(ledger[0]['resolution_status'],'diagnosis_unsupported')
        self.assertEqual(ledger[0]['precheck_status'],'diagnosis_unsupported')
        self.assertEqual(ledger[0]['resolution_history'][0]['cycle'],0)
        self.assertIsNone(ledger[0]['resolution_history'][0]['repair_receipt'])
        self.assertEqual(ledger[0]['precheck_receipt']['stage'],'precheck')
        self.assertFalse(ledger[0]['unresolved'])
        body=json.loads(call.call_args.args[0]);item=body['items']['1']
        self.assertEqual(item['original_chinese'],item['current_chinese'])
        self.assertEqual(body['current_draft_sha256'],a.fingerprint([asdict(row) for row in self.draft]))

    def test_mixed_candidates_only_confirmed_issue_editable(self):
        final,ledger,call=self.recall([a.diagnosis('ISSUE','ISSUE'),a.resolution('persists','diagnosis_unsupported'),
            a.encoded({'1':'改甲。'}),a.resolution('resolved','diagnosis_unsupported')])
        repairs=[item for item in call.call_args_list if item.args[2].stage.endswith('_repair')]
        self.assertEqual(len(repairs),1)
        items=json.loads(repairs[0].args[0])['items'];self.assertEqual(set(items),{'1'})
        self.assertEqual(items['1']['global_id'],1);self.assertEqual(items['1']['previous_resolution']['status'],'persists')
        self.assertEqual(final[1],self.draft[1]);self.assertEqual([row['repair_attempts'] for row in ledger],[1,0])
        post=call.call_args_list[-1];self.assertEqual(set(json.loads(post.args[0])['items']),{'1','2'})
        self.assertEqual(len(ledger[1]['resolution_history']),2)

    def test_withdrawn_candidate_can_reopen_after_other_edits(self):
        final,ledger,call=self.recall([a.diagnosis('ISSUE','ISSUE'),a.resolution('diagnosis_unsupported','persists'),
            a.encoded({'1':'改乙。'}),a.resolution('persists','resolved'),
            a.encoded({'1':'改甲。'}),a.resolution('resolved','resolved')])
        repairs=[json.loads(item.args[0])['items'] for item in call.call_args_list if item.args[2].stage.endswith('_repair')]
        self.assertEqual([item['1']['global_id'] for item in repairs],[2,1])
        self.assertEqual([row['repair_attempts'] for row in ledger],[1,1])
        self.assertEqual([item['status'] for item in ledger[0]['resolution_history']],['diagnosis_unsupported','persists','resolved'])
        self.assertTrue(all(not row['unresolved'] for row in ledger))
        final_hash=a.fingerprint([asdict(row) for row in final])
        self.assertTrue(all(row['resolution_history'][-1]['assembled_draft_sha256']==final_hash for row in ledger))

    def test_all_candidates_withdrawn_no_edit_and_final_states(self):
        final,ledger,call=self.recall([a.diagnosis('ISSUE','ISSUE'),a.resolution('diagnosis_unsupported','diagnosis_unsupported')])
        self.assertEqual(final,self.draft);self.assertEqual(call.call_count,2)
        self.assertTrue(all(row['repair_attempts']==0 and not row['changed'] for row in ledger))
        run=next(iter(self.saved()['runs'].values()))
        self.assertTrue(run['complete']);self.assertTrue(run['local_resolution_pass'])
        self.assertFalse(run['independently_evaluated']);self.assertEqual(run['unresolved_issue_ids'],[])
        self.assertEqual(run['initial_precheck_draft_sha256'],run['final_draft_sha256'])
        self.assertEqual(set(run['initial_precheck_results']),{'1','2'})

    def test_resolved_precheck_rejected_before_any_edit(self):
        with self.assertRaisesRegex(a.CoherenceError,'unchecked'):
            self.recall([a.diagnosis('ISSUE','OK'),a.resolution('resolved'),a.resolution('resolved')])
        run=next(iter(self.saved()['runs'].values()))
        self.assertFalse(run['complete'])
        self.assertTrue(all(not entry['request']['stage'].endswith('_repair') for entry in run['batches'].values()))
        self.assertEqual(self.draft[0].text,'甲句。')

    def test_missing_precheck_id_rejected_before_any_edit(self):
        with self.assertRaises(a.CoherenceError):
            self.recall([a.diagnosis('ISSUE','ISSUE'),a.resolution('persists'),a.resolution('persists')])
        run=next(iter(self.saved()['runs'].values()))
        self.assertFalse(run['complete'])
        self.assertTrue(all(not entry['request']['stage'].endswith('_repair') for entry in run['batches'].values()))

    def test_policy_separates_cached_diagnosis_and_checks(self):
        self.invoke([a.diagnosis('ISSUE','OK'),a.encoded({'1':'改甲。'}),a.resolution('resolved')])
        final,ledger,call=self.recall([a.diagnosis('ISSUE','OK'),a.resolution('diagnosis_unsupported')])
        self.assertEqual(call.call_count,2);self.assertEqual(final,self.draft)
        runs=self.saved()['runs'];self.assertEqual(len(runs),2)
        self.assertEqual({run['binding']['diagnosis_policy'] for run in runs.values()},{'conservative','recall-precheck'})
        for item in call.call_args_list:
            self.assertEqual(json.loads(item.args[0])['diagnosis_policy'],'recall-precheck')
        self.assertTrue(all(row['diagnosis_policy']=='recall-precheck' for row in ledger))

    def test_reopened_candidates_still_obey_three_global_repair_cycles(self):
        answers=[a.diagnosis('ISSUE','ISSUE'),a.resolution('diagnosis_unsupported','persists'),
            a.encoded({'1':'乙句。'}),a.resolution('persists','persists')]
        for _ in range(2):answers += [a.encoded({'1':'甲句。','2':'乙句。'}),a.resolution('persists','persists')]
        final,ledger,call=self.recall(answers)
        self.assertEqual(call.call_count,8);self.assertEqual(final,self.draft)
        self.assertEqual([row['repair_attempts'] for row in ledger],[2,3])
        self.assertTrue(all(row['unresolved'] for row in ledger))
        self.assertEqual([item['cycle'] for item in ledger[0]['resolution_history']],[0,1,2,3])

    def test_no_candidates_needs_no_precheck(self):
        final,ledger,call=self.recall([a.diagnosis('OK','OK')])
        self.assertEqual(call.call_count,1);self.assertEqual(final,self.draft)
        self.assertTrue(all(row['precheck_status']=='not_run' and not row['resolution_history'] for row in ledger))

    def test_invalid_policy_rejected_before_query(self):
        for value in ('unsupported',True,{},None):
            with self.subTest(value=value),patch('src.coherence_refine.call_llm') as call:
                with self.assertRaises(ValueError):
                    R.refine_resolution(self.source,self.draft,self.config,self.cache,diagnosis_policy=value)
                call.assert_not_called()
        self.capacity.assert_not_called()

    def test_cached_precheck_reparsed_and_checked_again_after_bad_hash(self):
        answers=[a.diagnosis('ISSUE','OK'),a.resolution('diagnosis_unsupported')]
        self.recall(answers)
        _,_,call=self.recall([]);call.assert_not_called()
        saved=self.saved();run=next(iter(saved['runs'].values()))
        entry=next(item for item in run['batches'].values() if item['request']['stage'].endswith('_precheck'))
        entry['response_sha256']='bad'
        self.cache.write_text(a.encoded(saved),encoding='utf-8')
        final,ledger,call=self.recall([a.resolution('diagnosis_unsupported')])
        self.assertEqual(call.call_count,1);self.assertEqual(final,self.draft)
        self.assertEqual(ledger[0]['precheck_receipt']['stage'],'precheck')

if __name__=='__main__':unittest.main(verbosity=2)
