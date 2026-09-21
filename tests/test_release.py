"""Release behavior: a completed writer run is not an independent signoff."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.release import (ALLOWED_REVIEWER_MODELS, BENCHMARK_VERSION, CHECKS, COHERENCE_CHECKS,
                         REVIEWER_SCOPE, SOURCE_VERIFIED, TARGET_COHERENCE, WRITER_MODEL,
                         coherence_warnings, prepare_assessment, release_unit,
                         validate_assessment, validate_coherence_assessment)
from src.workflow_state import read_json, write_json

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.unit=self.root/'work';final=self.unit/'final';final.mkdir(parents=True)
        (final/'episode.mp4').write_bytes(b'test media')
        self.source=final/'episode.srt';self.target=final/'episode.zh.srt'
        self.source.write_text('1\n00:00:01,000 --> 00:00:03,000\n来るな。\n')
        self.target.write_text('1\n00:00:01,000 --> 00:00:03,000\n别过来。\n')
        self.path=self.unit/'review/release-assessment.json'
        self.record=prepare_assessment(self.unit,self.path)
    def approve(self):
        self.record.update(score=6,reviewer=dict(name='Independent fixture reviewer',kind='independent_model',
                               model='gpt-6-astra',scope=REVIEWER_SCOPE,feedback_to_writer='score_only',independent_of_writer=True),
                           reviewed_at='2026-09-13T15:00:00+08:00',checks={name:True for name in CHECKS})
        write_json(self.path,self.record)
    def test_trial_template_records_actual_writer_without_approving(self):
        for alias in ('qwen3.8-27b-q8-trial', 'qwen3.8-27b-orca'):
            with self.subTest(alias=alias):
                path = self.path.with_name(alias+'-assessment.json')
                record = prepare_assessment(self.unit,path,writer_model=alias)
                self.assertEqual(record['writer_model'],alias)
                self.assertIsNone(record['score'])
                self.assertTrue(validate_assessment(self.unit,record))
                self.approve()
                self.record['writer_model']=alias
                self.assertEqual(validate_assessment(self.unit,self.record),[])
        with self.assertRaisesRegex(ValueError,'local writer model'):
            prepare_assessment(self.unit,self.path.with_name('bad.json'),writer_model='gpt-6-astra')

    def test_cli_cannot_relabel_an_existing_review_during_check(self):
        from main import build_parser, cmd_release
        old = self.path.read_bytes()
        args = build_parser().parse_args(['release',str(self.unit),'--check',
            '--writer-model','qwen3.8-27b-q8-trial'])
        self.assertEqual(cmd_release(args),1)
        self.assertEqual(self.path.read_bytes(),old)

    def test_prepared_template_never_approves_output(self):
        self.assertTrue(validate_assessment(self.unit,self.record))
        with self.assertRaisesRegex(ValueError,'Release blocked'):release_unit(self.unit,self.path,self.root/'released')
        self.assertFalse((self.root/'released').exists())
    def test_release_requires_independence_all_checks_and_no_open_issues(self):
        self.approve();self.assertEqual(validate_assessment(self.unit,self.record),[])
        self.record['reviewer']['independent_of_writer']=False
        self.assertTrue(validate_assessment(self.unit,self.record))
        self.record['reviewer']['independent_of_writer']=True
        self.record['checks']['source_checked_in_full']=False
        self.assertTrue(validate_assessment(self.unit,self.record))
        self.record['checks']['source_checked_in_full']=True
        self.record['open_issues']=['Negation still wrong']
        self.assertTrue(validate_assessment(self.unit,self.record))
    def test_edits_extra_files_and_removed_files_invalidate_signoff(self):
        self.approve();original=self.target.read_bytes()
        self.target.write_text(self.target.read_text().replace('别过来','过来'))
        self.assertTrue(any('changed' in e for e in validate_assessment(self.unit,self.record)))
        self.target.write_bytes(original)
        extra=self.unit/'final/extra.txt';extra.write_text('new')
        self.assertTrue(validate_assessment(self.unit,self.record));extra.unlink()
        self.target.unlink();self.assertTrue(validate_assessment(self.unit,self.record))
    def test_valid_release_preserves_review_folder_and_refuses_overwrite(self):
        self.approve();dest=release_unit(self.unit,self.path,self.root/'released')
        self.assertTrue((dest/'final/episode.zh.srt').is_file())
        self.assertTrue(dest.with_suffix('.zip').is_file());self.assertTrue(self.unit.is_dir())
        with self.assertRaises(FileExistsError):release_unit(self.unit,self.path,self.root/'released')
    def test_copy_change_cannot_be_packaged_with_stale_approval(self):
        import shutil
        self.approve();original=shutil.copytree
        def corrupt(src,dst,*args,**kwargs):
            result=original(src,dst,*args,**kwargs)
            (Path(dst)/'episode.zh.srt').write_text('corrupted')
            return result
        with patch('src.release.shutil.copytree',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'changed while packaging'):release_unit(self.unit,self.path,self.root/'released')
        self.assertFalse((self.root/'released/work').exists())
    def test_structural_errors_cannot_be_waived(self):
        self.target.write_text('1\n00:00:03,000 --> 00:00:01,000\n别过来。\n')
        self.path.unlink();self.record=prepare_assessment(self.unit,self.path);self.approve()
        self.record['finding_exceptions']={f['id']:'Ignore this' for f in self.record['automatic_findings']}
        self.assertTrue(validate_assessment(self.unit,self.record))


    def reassess(self):
        self.path.unlink()
        self.record = prepare_assessment(self.unit, self.path)
        self.approve()

    def waive_findings(self):
        self.record['finding_exceptions'] = {
            finding['id']: 'Verified intentional exception'
            for finding in self.record['automatic_findings']
        }

    def test_malformed_srt_chunks_cannot_be_silently_skipped(self):
        valid = self.target.read_text()
        for extra in ('garbage\n', '2\n00:00:04,000 --> 00:00:05,000\n',
                      'unindexed\n00:00:04,000 --> 00:00:05,000\nLost cue\n'):
            with self.subTest(extra=extra):
                self.target.write_text(valid + '\n' + extra)
                self.reassess()
                self.waive_findings()
                errors = validate_assessment(self.unit, self.record)
                self.assertTrue(any('invalid-srt' in error for error in errors))

    def test_timestamp_ranges_and_encoding_are_validated_without_crashing(self):
        for content in (b'1\n00:60:01,000 --> 00:60:03,000\ntext\n',
                        b'1\n00:00:61,000 --> 00:01:03,000\ntext\n',
                        b'1\n00:00:01,000 --> 00:00:03,000\n\xff\n'):
            with self.subTest(content=content):
                self.target.write_bytes(content)
                self.reassess()
                self.waive_findings()
                self.assertTrue(any('invalid-srt' in error for error in
                                    validate_assessment(self.unit, self.record)))

    def test_sidecars_and_other_media_sources_do_not_count_as_translation(self):
        self.target.unlink()
        for name in ('episode.utterances.srt', 'episode.pre-review.srt',
                     'episode.pre-review.zh.srt'):
            (self.unit / 'final' / name).write_bytes(self.source.read_bytes())
        # "fr" is a valid language code but this file belongs to other media.
        (self.unit / 'final/episode.fr.mp4').write_bytes(b'other media')
        (self.unit / 'final/episode.fr.srt').write_bytes(self.source.read_bytes())
        (self.unit / 'final/episode.fr.zh.srt').write_bytes(self.source.read_bytes())
        self.reassess()
        self.waive_findings()
        errors = validate_assessment(self.unit, self.record)
        self.assertTrue(any('final/episode.mp4:0:missing-translation' in error for error in errors))
        self.assertTrue(any('orphan-subtitles' in error for error in errors))

    def test_missing_source_cannot_be_waived_as_empty_track(self):
        self.source.unlink()
        self.reassess()
        self.waive_findings()
        self.assertTrue(any('missing-source' in error for error in
                            validate_assessment(self.unit, self.record)))

    def test_verified_empty_track_and_brief_cue_can_have_reviewed_exceptions(self):
        (self.unit / 'final/rest.wav').write_bytes(b'verified rest media')
        for path in (self.source, self.target):
            path.write_text(path.read_text().replace('00:00:03,000', '00:00:01,300'))
        self.reassess()
        self.assertTrue(validate_assessment(self.unit, self.record))
        self.waive_findings()
        self.assertEqual(validate_assessment(self.unit, self.record), [])

    def test_symlinked_final_directory_and_removed_subtitles_invalidate_approval(self):
        self.approve()
        final = self.unit / 'final'
        actual = self.root / 'actual-final'
        final.rename(actual)
        final.symlink_to(actual, target_is_directory=True)
        self.assertTrue(any('symlink' in error for error in
                            validate_assessment(self.unit, self.record)))
        final.unlink()
        actual.rename(final)
        self.source.unlink()
        self.target.unlink()
        self.assertTrue(validate_assessment(self.unit, self.record))

    def test_release_destination_cannot_recursively_copy_into_source_final(self):
        self.approve()
        destination = self.unit / 'final/released'
        with self.assertRaisesRegex(ValueError, 'inside the source final'):
            release_unit(self.unit, self.path, destination)
        self.assertFalse(destination.exists())

    def test_regional_language_code_is_a_valid_translation(self):
        self.target.rename(self.unit / 'final/episode.zh-Hant.srt')
        self.reassess()
        self.assertEqual(validate_assessment(self.unit, self.record), [])

    def test_invalid_assessment_types_and_scores_fail_without_crashing(self):
        self.approve()
        for score in (True, float('nan'), float('inf'), 10**500, -1, 5.99, 11, '6'):
            with self.subTest(score=score):
                self.record['score'] = score
                self.assertTrue(validate_assessment(self.unit, self.record))
        self.record['score'] = 6
        self.record['reviewer']['kind'] = ['human']
        self.assertTrue(validate_assessment(self.unit, self.record))

    def test_assessment_cannot_be_part_of_its_own_deliverable_manifest(self):
        path = self.unit / 'final/release-assessment.json'
        with self.assertRaisesRegex(ValueError, 'outside final'):
            prepare_assessment(self.unit, path)
        self.assertFalse(path.exists())

    def test_even_one_millisecond_overlap_requires_a_repair(self):
        for path in (self.source, self.target):
            path.write_text(path.read_text() + '\n2\n00:00:02,999 --> 00:00:05,000\nNext line\n')
        self.reassess()
        self.waive_findings()
        self.assertTrue(any('overlap' in error for error in
                            validate_assessment(self.unit, self.record)))

    def test_template_records_writer_and_evaluation_policy_without_approval(self):
        self.assertEqual(self.record['benchmark'], BENCHMARK_VERSION)
        self.assertEqual(self.record['writer_model'], WRITER_MODEL)
        self.assertEqual(self.record['reviewer']['kind'], 'independent_model')
        self.assertEqual(self.record['reviewer']['scope'], 'evaluation_only')
        self.assertEqual(self.record['reviewer']['feedback_to_writer'], 'score_only')
        self.assertEqual(self.record['reviewer']['model'], '')
        self.assertFalse(self.record['reviewer']['independent_of_writer'])
        self.assertIsNone(self.record['score'])
        self.assertTrue(all(value is False for value in self.record['checks'].values()))

    def test_each_allowed_stronger_model_still_requires_full_validation(self):
        self.approve()
        for model in ALLOWED_REVIEWER_MODELS:
            with self.subTest(model=model):
                self.record['reviewer']['model'] = model
                self.assertEqual(validate_assessment(self.unit, self.record), [])
                self.record['checks']['source_checked_in_full'] = False
                self.assertTrue(validate_assessment(self.unit, self.record))
                self.record['checks']['source_checked_in_full'] = True
                self.record['open_issues'] = ['Known source uncertainty']
                self.assertTrue(validate_assessment(self.unit, self.record))
                self.record['open_issues'] = []

    def test_qwen_cannot_approve_release_by_claiming_independence_or_a_stronger_name(self):
        self.approve()
        self.record['reviewer']['name'] = 'GPT-6 Astra reviewer'
        self.record['reviewer']['model'] = WRITER_MODEL
        self.assertTrue(self.record['reviewer']['independent_of_writer'])
        errors = validate_assessment(self.unit, self.record)
        self.assertTrue(any('reviewer model' in error for error in errors))

    def test_human_review_is_supplemental_not_the_required_model_check(self):
        self.approve()
        self.record['reviewer']['kind'] = 'human'
        self.assertTrue(any('independent model' in error for error in
                            validate_assessment(self.unit, self.record)))

    def test_stronger_model_must_only_evaluate_and_cannot_be_the_writer(self):
        self.approve()
        for scope in ('translation', 'repair', '', None):
            with self.subTest(scope=scope):
                self.record['reviewer']['scope'] = scope
                self.assertTrue(any('evaluation_only' in error for error in
                                    validate_assessment(self.unit, self.record)))
        self.record['reviewer']['scope'] = REVIEWER_SCOPE
        self.record['writer_model'] = self.record['reviewer']['model']
        self.assertTrue(any('writer model' in error for error in
                            validate_assessment(self.unit, self.record)))

    def test_audit_findings_cannot_be_attested_as_writer_feedback(self):
        self.approve()
        for feedback in (None, '', 'findings', 'categories_only', 'replacement_text', True):
            with self.subTest(feedback=feedback):
                self.record['reviewer']['feedback_to_writer'] = feedback
                self.assertTrue(any('score_only' in error for error in
                                    validate_assessment(self.unit, self.record)))
        self.record['reviewer'].pop('feedback_to_writer')
        self.assertTrue(any('score_only' in error for error in
                            validate_assessment(self.unit, self.record)))
        self.record['reviewer']['feedback_to_writer'] = 'score_only'
        self.assertEqual(validate_assessment(self.unit, self.record), [])

    def test_missing_or_malformed_model_identity_cannot_pass(self):
        self.approve()
        for model in (None, '', ['gpt-6-astra'], {'model': 'gpt-6-astra'}, 'another-model'):
            with self.subTest(model=model):
                self.record['reviewer']['model'] = model
                self.assertTrue(validate_assessment(self.unit, self.record))
        self.record['reviewer']['model'] = 'gpt-6-astra'
        self.record.pop('writer_model')
        self.assertTrue(validate_assessment(self.unit, self.record))

    def test_version_one_assessment_requires_a_new_unapproved_record(self):
        self.approve()
        self.record['benchmark'] = 'subtitle-quality-v1'
        write_json(self.path, self.record)
        self.assertTrue(any('outdated benchmark' in error for error in
                            validate_assessment(self.unit, self.record)))
        old_bytes = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_assessment(self.unit, self.path)
        replacement = self.path.with_name('release-assessment-v3.json')
        current = prepare_assessment(self.unit, replacement)
        self.assertEqual(self.path.read_bytes(), old_bytes)
        self.assertEqual(current['benchmark'], BENCHMARK_VERSION)
        self.assertIsNone(current['score'])
        self.assertEqual(current['reviewer']['model'], '')

    def prepare_coherence(self):
        self.path = self.path.with_name('coherence-assessment.json')
        self.record = prepare_assessment(self.unit, self.path,
                                         assessment_scope=TARGET_COHERENCE)

    def approve_coherence(self):
        self.approve()
        self.record.update(score=4, checks={name: False for name in CHECKS},
                           coherence_checks={name: True for name in COHERENCE_CHECKS})
        write_json(self.path, self.record)

    def test_coherence_template_is_explicit_unapproved_and_cannot_release(self):
        self.prepare_coherence()
        self.assertEqual(self.record['assessment_scope'], TARGET_COHERENCE)
        self.assertEqual(self.record['writer_execution'], 'local')
        self.assertIsNone(self.record['score'])
        self.assertTrue(all(value is False for value in self.record['coherence_checks'].values()))
        self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        self.approve_coherence()
        self.record['open_issues'] = ['Source-aware accuracy remains unverified']
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, 'Release blocked'):
            release_unit(self.unit, self.path, self.root/'released')
        self.assertFalse((self.root/'released').exists())

    def test_each_coherence_check_and_issue_must_pass_independently(self):
        self.prepare_coherence()
        self.approve_coherence()
        for name in COHERENCE_CHECKS:
            with self.subTest(check=name):
                self.record['coherence_checks'][name] = False
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
                self.record['coherence_checks'][name] = True
        for issues in (['Confusing Chinese context'], None, {}, ''):
            with self.subTest(issues=issues):
                self.record['coherence_open_issues'] = issues
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        self.record['coherence_open_issues'] = []
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])

    def test_coherence_can_retain_explicit_non_obvious_polish_notes(self):
        self.prepare_coherence()
        self.approve_coherence()
        self.record['coherence_polish_notes'] = [
            {'severity': 'non_obvious_polish', 'note': 'Minor polish visible only on careful review'}]
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
        self.assertEqual(len(self.record['coherence_polish_notes']), 1)

    def test_older_v3_records_without_polish_notes_keep_existing_behavior(self):
        self.prepare_coherence()
        self.approve_coherence()
        self.record.pop('coherence_polish_notes', None)
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
        self.record.update(assessment_scope=SOURCE_VERIFIED, score=6,
                           checks={name: True for name in CHECKS})
        self.assertEqual(validate_assessment(self.unit, self.record), [])

    def test_polish_notes_cannot_waive_obvious_issues_or_failed_checks(self):
        self.prepare_coherence()
        self.approve_coherence()
        self.record['coherence_polish_notes'] = [
            {'severity': 'non_obvious_polish', 'note': 'Minor polish only'}]
        for issue in ('Obvious confusion', 'Contradictory Chinese', 'Unclear actor or action'):
            with self.subTest(issue=issue):
                self.record['coherence_open_issues'] = [issue]
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        self.record['coherence_open_issues'] = []
        for check in COHERENCE_CHECKS:
            with self.subTest(check=check):
                self.record['coherence_checks'][check] = False
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
                self.record['coherence_checks'][check] = True

    def test_polish_notes_require_explicit_valid_minor_classification(self):
        self.prepare_coherence()
        self.approve_coherence()
        valid_note = {'severity': 'non_obvious_polish', 'note': 'Minor polish only'}
        invalid = (None, '', {}, ['A bare note has no classification'], [None],
                   [{'note': 'Missing classification'}],
                   [{'severity': 'obvious', 'note': 'An obvious error'}],
                   [{'severity': 'non_obvious_polish', 'note': ''}],
                   [{'severity': 'non_obvious_polish', 'note': '   '}],
                   [{'severity': 'non_obvious_polish', 'note': 4}],
                   [{**valid_note, 'waive_checks': True}])
        for notes in invalid:
            with self.subTest(notes=notes):
                self.record['coherence_polish_notes'] = notes
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))

    def test_polish_notes_cannot_be_carried_into_a_score_six_release(self):
        self.approve()
        self.record['coherence_polish_notes'] = [
            {'severity': 'non_obvious_polish', 'note': 'A recorded minor language issue remains'}]
        self.assertTrue(any('target-language issues' in error for error in
                            validate_assessment(self.unit, self.record)))
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, 'Release blocked'):
            release_unit(self.unit, self.path, self.root/'released')
        self.assertFalse((self.root/'released').exists())

    def test_coherence_cannot_substitute_for_source_verified_review(self):
        self.prepare_coherence()
        self.approve_coherence()
        self.record['score'] = 6
        self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        self.assertTrue(any('scope' in error for error in validate_assessment(self.unit, self.record)))
        self.record['assessment_scope'] = SOURCE_VERIFIED
        self.assertTrue(any('full-source' in error for error in validate_assessment(self.unit, self.record)))
        self.record['checks'] = {name: True for name in CHECKS}
        self.record['open_issues'] = ['A source-aware error remains']
        self.assertTrue(validate_assessment(self.unit, self.record))
        self.record['open_issues'] = []
        self.assertEqual(validate_assessment(self.unit, self.record), [])
        self.record['coherence_open_issues'] = ['A known contradiction cannot be hidden']
        self.assertTrue(validate_assessment(self.unit, self.record))

    def test_coherence_requires_current_scope_and_valid_score(self):
        self.prepare_coherence()
        self.approve_coherence()
        for score in (True, float('nan'), float('inf'), 10**500, -1, 3.99, 6, 11, '4'):
            with self.subTest(score=score):
                self.record['score'] = score
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        self.record['score'] = 5
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
        self.record.pop('assessment_scope')
        self.assertTrue(validate_coherence_assessment(self.unit, self.record))
        with self.assertRaisesRegex(ValueError, 'Assessment scope'):
            prepare_assessment(self.unit, self.path.with_name('bad-scope.json'), assessment_scope='release_anyway')

    def test_duration_findings_warn_at_four_but_still_block_six(self):
        for number, timestamp in enumerate(('00:00:01,000 --> 00:00:01,080',
                                            '00:00:01,000 --> 00:00:12,000')):
            with self.subTest(timestamp=timestamp):
                for path in (self.source, self.target):
                    lines = path.read_text().splitlines()
                    lines[1] = timestamp
                    path.write_text('\n'.join(lines)+'\n')
                self.path = self.path.with_name(f'timing-{number}.json')
                self.record = prepare_assessment(self.unit, self.path, assessment_scope=TARGET_COHERENCE)
                self.approve_coherence()
                self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
                warnings = coherence_warnings(self.unit)
                self.assertEqual(len(warnings), 2)
                self.assertTrue(all(row['kind'] in {'short-cue', 'long-cue'} for row in warnings))
                self.record.update(assessment_scope=SOURCE_VERIFIED, score=6,
                                   checks={name: True for name in CHECKS})
                self.assertTrue(any('cue' in error for error in validate_assessment(self.unit, self.record)))

    def test_broken_structure_cannot_be_waived_at_four(self):
        self.target.write_text('1\n00:00:03,000 --> 00:00:01,000\n别过来。\n')
        self.prepare_coherence()
        self.approve_coherence()
        self.waive_findings()
        self.assertTrue(any('nonpositive' in error or 'invalid-srt' in error
                            for error in validate_coherence_assessment(self.unit, self.record)))

    def test_final_edits_invalidate_coherence_approval(self):
        self.prepare_coherence()
        self.approve_coherence()
        self.assertEqual(validate_coherence_assessment(self.unit, self.record), [])
        self.target.write_text(self.target.read_text().replace('别过来', '过来'))
        self.assertTrue(any('changed' in error for error in
                            validate_coherence_assessment(self.unit, self.record)))

    def test_any_identified_local_writer_is_allowed_but_evaluators_are_not(self):
        for model in ('local-specialist-translator-q4', '/models/local-writer.gguf'):
            with self.subTest(model=model):
                record = prepare_assessment(self.unit, self.path.with_name(str(len(model))+'.json'),
                                            writer_model=model)
                self.approve()
                self.record['writer_model'] = record['writer_model']
                self.assertEqual(validate_assessment(self.unit, self.record), [])
        for model in (None, '', '  ', [], 'gpt-6-astra', 'FABLE-5.1'):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, 'local writer model'):
                    prepare_assessment(self.unit, self.path.with_name('bad-writer.json'), writer_model=model)
        for execution in ('remote', '', None, True):
            with self.subTest(execution=execution):
                self.record['writer_execution'] = execution
                self.assertTrue(validate_assessment(self.unit, self.record))

    def test_coherence_requires_the_same_independent_score_only_reviewer_policy(self):
        self.prepare_coherence()
        self.approve_coherence()
        for field, value in (('model', WRITER_MODEL), ('independent_of_writer', False),
                             ('scope', 'repair'), ('feedback_to_writer', 'categories_only')):
            with self.subTest(field=field):
                original = self.record['reviewer'][field]
                self.record['reviewer'][field] = value
                self.assertTrue(validate_coherence_assessment(self.unit, self.record))
                self.record['reviewer'][field] = original
        self.record['writer_execution'] = 'remote'
        self.assertTrue(validate_coherence_assessment(self.unit, self.record))

    def test_v2_history_is_preserved_and_needs_a_new_v3_review(self):
        self.approve()
        self.record['benchmark'] = 'subtitle-quality-v2'
        self.record['score'] = 2
        write_json(self.path, self.record)
        historical = self.path.read_bytes()
        self.assertTrue(validate_assessment(self.unit, self.record))
        self.prepare_coherence()
        self.assertEqual(self.record['benchmark'], 'subtitle-quality-v3')
        self.assertIsNone(self.record['score'])
        self.assertEqual((self.path.parent/'release-assessment.json').read_bytes(), historical)

    def test_cli_prepares_and_checks_coherence_for_custom_local_writer(self):
        from main import build_parser, cmd_release
        parser = build_parser()
        self.path = self.path.with_name('cli-coherence.json')
        common = ['release', str(self.unit), '--assessment', str(self.path),
                  '--scope', TARGET_COHERENCE]
        args = parser.parse_args(common+['--prepare', '--writer-model', 'local-specialist-7b'])
        with self.assertLogs('auto-subtitle', level='INFO'):
            self.assertEqual(cmd_release(args), 0)
        self.record = read_json(self.path)
        self.assertEqual(self.record['writer_model'], 'local-specialist-7b')
        self.assertEqual(self.record['assessment_scope'], TARGET_COHERENCE)
        self.assertIsNone(self.record['score'])
        with self.assertLogs('auto-subtitle', level='ERROR'):
            self.assertEqual(cmd_release(parser.parse_args(common+['--check'])), 1)
        self.approve_coherence()
        approved = self.path.read_bytes()
        with self.assertLogs('auto-subtitle', level='INFO') as messages:
            self.assertEqual(cmd_release(parser.parse_args(common+['--check'])), 0)
        self.assertTrue(any('source fidelity is not certified' in line for line in messages.output))
        self.assertEqual(self.path.read_bytes(), approved)

    def test_cli_scope_mismatch_cannot_relabel_coherence_as_release(self):
        from main import build_parser, cmd_release
        self.prepare_coherence()
        self.approve_coherence()
        approved = self.path.read_bytes()
        args = build_parser().parse_args(['release', str(self.unit), '--assessment', str(self.path), '--check'])
        self.assertEqual(args.scope, SOURCE_VERIFIED)
        with self.assertLogs('auto-subtitle', level='ERROR') as messages:
            self.assertEqual(cmd_release(args), 1)
        self.assertTrue(any('scope must be source_verified' in line for line in messages.output))
        self.assertEqual(self.path.read_bytes(), approved)

    def test_cli_coherence_packaging_is_refused_before_release_call(self):
        from main import build_parser, cmd_release
        self.prepare_coherence()
        self.approve_coherence()
        destination = self.root/'should-not-release'
        args = build_parser().parse_args(['release', str(self.unit), '--assessment', str(self.path),
                                         '--scope', TARGET_COHERENCE, '--destination', str(destination)])
        with patch('src.release.release_unit') as package:
            with self.assertLogs('auto-subtitle', level='ERROR') as messages:
                self.assertEqual(cmd_release(args), 1)
            package.assert_not_called()
        self.assertTrue(any('score-four milestone' in line for line in messages.output))
        self.assertFalse(destination.exists())

    def test_cli_coherence_reports_duration_warnings_without_claiming_playback(self):
        from main import build_parser, cmd_release
        for path in (self.source, self.target):
            path.write_text(path.read_text().replace('00:00:03,000', '00:00:01,080'))
        self.prepare_coherence()
        self.approve_coherence()
        args = build_parser().parse_args(['release', str(self.unit), '--assessment', str(self.path),
                                         '--scope', TARGET_COHERENCE, '--check'])
        with self.assertLogs('auto-subtitle', level='INFO') as messages:
            self.assertEqual(cmd_release(args), 0)
        self.assertEqual(sum('Timing requires separate review' in line for line in messages.output), 2)
        self.assertTrue(any('source fidelity is not certified' in line for line in messages.output))

    def test_cli_explicit_empty_writer_is_not_silently_replaced_with_default(self):
        from main import build_parser, cmd_release
        parser = build_parser()
        path = self.path.with_name('empty-writer.json')
        args = parser.parse_args(['release', str(self.unit), '--prepare', '--writer-model', '',
                                  '--assessment', str(path), '--scope', TARGET_COHERENCE])
        with self.assertLogs('auto-subtitle', level='ERROR'):
            self.assertEqual(cmd_release(args), 1)
        self.assertFalse(path.exists())
        original = self.path.read_bytes()
        args = parser.parse_args(['release', str(self.unit), '--check', '--writer-model', ''])
        with self.assertLogs('auto-subtitle', level='ERROR') as messages:
            self.assertEqual(cmd_release(args), 1)
        self.assertTrue(any('only for preparing a new assessment' in line for line in messages.output))
        self.assertEqual(self.path.read_bytes(), original)

if __name__=='__main__':unittest.main()
