"""Behavioral tests for local repair transactions; no release approval is implied."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src import quality, repair_verification as rv
from src.config import TranslateConfig
from src.evidence import CueEvidence
from src.translate import SrtBlock


def verdict(ids, failed=None):
    return json.dumps({str(i): {**{c: ('no' if i == failed and c == 'no_cross_row_regression' else 'yes') for c in rv.CHECKS},
                               'reason': 'The selected source supports this local change.'} for i in ids})


class RepairTransactions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / 'verify.json'
        self.cfg = TranslateConfig(scene_context_lines=1, delta_verification=True)
        self.source = [SrtBlock(i, f'00:00:{i:02d},000 --> 00:00:{i:02d},900', f'source {i}') for i in range(1, 11)]
        self.draft = [replace(b, text=f'before {b.index}') for b in self.source]
        self.proposed = [replace(b) for b in self.draft]
        self.ledger = [{'line': b.index, 'status': 'unresolved', 'reason': 'Existing independent problem',
                        'source_decision': {'status': 'unresolved', 'reason': 'Unclear audio'}} for b in self.source]

    def run_verifier(self, changed, response):
        for ln in changed:
            self.proposed[ln - 1].text = f'after {ln}'
        with patch.object(rv, 'call_llm', return_value=response) as llm:
            final, ledger = rv.verify_repairs(self.source, self.draft, self.proposed, self.ledger, changed, self.cfg, self.cache)
        return final, ledger, llm

    def test_preexisting_neighbor_issue_does_not_veto_supported_edit(self):
        final, ledger, llm = self.run_verifier([2], verdict([1]))
        self.assertEqual(final[1].text, 'after 2')
        self.assertEqual(ledger[1]['status'], 'corrected')
        self.assertEqual(ledger[0]['status'], 'unresolved')
        self.assertEqual(ledger[1]['source_decision']['status'], 'unresolved')
        self.assertIn('Existing independent problem', llm.call_args.args[0])
        self.assertEqual(self.draft[1].text, 'before 2')

    def test_new_cross_row_error_reverts_entire_dependency_component(self):
        final, ledger, _ = self.run_verifier([2, 4], verdict([1, 2], failed=2))
        self.assertEqual([b.text for b in final], [b.text for b in self.draft])
        self.assertTrue(all(ledger[i-1]['status'] == 'unresolved' for i in (2,4)))
        self.assertEqual(ledger[1]['repair_verification']['component'], [2,4])
        self.assertEqual(ledger[1]['repair_candidate'], 'after 2')

    def test_separated_components_are_verified_in_the_committed_context(self):
        for ln in (2, 8):
            self.proposed[ln-1].text = f'after {ln}'
        with patch.object(rv, 'call_llm', side_effect=[verdict([1]), verdict([1], failed=1)]) as llm:
            final, ledger = rv.verify_repairs(self.source,self.draft,self.proposed,self.ledger,[2,8],self.cfg,self.cache)
        self.assertEqual(llm.call_count,2)
        self.assertEqual(final[1].text,'after 2')
        self.assertEqual(final[7].text,'before 8')
        self.assertNotIn('source 8', llm.call_args_list[0].args[0])
        self.assertNotIn('source 2', llm.call_args_list[1].args[0])
        self.assertNotEqual(ledger[1]['repair_verification']['input_sha256'],ledger[7]['repair_verification']['input_sha256'])

    def test_missing_or_uncertain_verdict_does_not_commit(self):
        for text, status in [('{}','unchecked'), (verdict([1]).replace('"yes"','"uncertain"',1),'unresolved')]:
            with self.subTest(text=text):
                self.cache.unlink(missing_ok=True)
                final, ledger, _=self.run_verifier([2],text)
                self.assertEqual(final[1].text,'before 2')
                self.assertEqual(ledger[1]['status'],status)

    def test_changed_neighbor_invalidates_cached_approval(self):
        self.run_verifier([2],verdict([1]))
        self.draft[0].text='Different neighbor'
        _,_,llm=self.run_verifier([2],verdict([1]))
        self.assertEqual(llm.call_count,1)

    def test_strict_schema_rejects_duplicates_and_nonboolean_verdicts(self):
        valid=verdict([1])
        self.assertEqual(set(rv.parse(valid,[1])),{1})
        bad=[valid.replace('"yes"','true',1), valid.replace('"1":','"9":',1), '{"1":{},"1":{}}', valid.replace('source supports','source\\nsupports')]
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(rv.parse(text,[1]),{})

    def test_source_blocker_survives_quality_integration(self):
        source=[SrtBlock(1,'00:00:01,000 --> 00:00:03,000','私が彼を起こします。')]
        draft=[replace(source[0],text='他会叫醒我。')]
        evidence=[CueEvidence(line=1,raw=source[0].text,grade='C')]
        decisions=[{'line':1,'status':'unresolved','candidate':'W','text':source[0].text,'reason':'Audio doubt'}]
        with patch.object(quality,'call_llm',side_effect=['[1] ISSUE|role|Actors are reversed','[1] FIX|我会叫醒他。']), patch.object(rv,'call_llm',return_value=verdict([1])):
            final,ledger=quality.selective_qa(source,draft,evidence,decisions,self.cfg,self.cache)
        self.assertEqual(final[0].text,'我会叫醒他。')
        self.assertEqual(ledger[0]['target_status'],'corrected')
        self.assertEqual(ledger[0]['status'],'unresolved')
        self.assertIn('Audio doubt',ledger[0]['reason'])

    def test_flag_is_opt_in_and_part_of_workflow_fingerprint(self):
        from main import build_parser
        from src.workflow import configuration_key
        self.assertFalse(build_parser().parse_args(['pipeline']).delta_verification)
        self.assertTrue(build_parser().parse_args(['pipeline','--delta-verification']).delta_verification)
        self.assertNotEqual(configuration_key(self.cfg),configuration_key(replace(self.cfg,delta_verification=False)))

if __name__=='__main__':
    unittest.main()
