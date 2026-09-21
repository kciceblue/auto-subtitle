"""Resume the fixed Q1/Q2/Q3 late-evidence campaign; explicit execution only.

No subtitle wording or local/external findings are printed. Scalar reviews are
stored outside local model inputs. This controller records eligible candidates
but never replaces output/selected. Interrupted dispatches cannot be rerolled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import contextual_evidence_review as er
from src import late_audio as audio
from src import revisit_workflow as rw
from src.quality import seconds
from src.translate import parse_srt
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'bounded-revisit-controller-1'
OWNER_COUNT = 66
ROUND_IDS = ('Q1', 'Q2', 'Q3')
LOG = logging.getLogger(__name__)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def owners_from_rows(rows) -> list[dict]:
    er.owner_rows_sha256(rows)
    require(len(rows) == OWNER_COUNT, 'Campaign must retain all 66 owners')
    return [{'id': r.index, 'start': seconds(r.ts_line.split(' --> ')[0]),
             'end': seconds(r.ts_line.split(' --> ')[1])} for r in rows]


def select_deep(diagnosis: dict) -> tuple[list[int], list[int]]:
    """Priority descending, original ID ascending; boundary views use raw audio.

    Selection uses categorical local diagnostics only, never an external score.
    At most 20 owners receive two views each and two scheduled ASR observers.
    """
    audits = diagnosis['audits']
    require(set(audits) == {str(i) for i in range(1, OWNER_COUNT + 1)}, 'Incomplete diagnosis')
    eligible = []
    for key, row in audits.items():
        require(type(row['priority']) is int and 0 <= row['priority'] <= 3
                and type(row['needs_audio']) is bool, 'Invalid audio priority')
        require(row['verdict'] in {'ok', 'source_uncertain', 'target_error', 'both'}
                and row['audio_problem'] in {'boundary', 'masking', 'overlap', 'unknown'},
                'Invalid audio hypothesis')
        if row['needs_audio'] or row['verdict'] in {'source_uncertain', 'both'}:
            eligible.append(int(key))
    chosen = sorted(eligible, key=lambda i: (-audits[str(i)]['priority'], i))[:20]
    # Stable original order is the native crop API's ownership order.
    chosen.sort()
    return chosen, [i for i in chosen if audits[str(i)]['audio_problem'] != 'boundary']


def candidate_gate(folder: Path, round_id: str) -> dict:
    path = folder / round_id
    candidate = rw.read(path / 'candidate.json')
    if candidate.get('status') == 'unchanged_no_actionable_questions':
        return candidate
    source, target, raw = rw.rows(path / 'source.json'), rw.rows(path / 'target.json'), rw.rows(folder / 'source.json')
    owners_from_rows(source); owners_from_rows(target)
    require([(r.index, r.ts_line) for r in source] == [(r.index, r.ts_line) for r in target]
            == [(r.index, r.ts_line) for r in raw], 'Candidate owner geometry changed')
    require(candidate.get('source_sha256') == file_hash(path / 'source.json')
            and candidate.get('target_sha256') == file_hash(path / 'target.json'), 'Candidate text artifact changed')
    checked = candidate.get('final_validation', {})
    require(set(checked) == {str(i) for i in range(1, OWNER_COUNT + 1)}, 'Missing final local owner validation')
    require(all(type(v.get('new_material_error')) is bool and type(v.get('pass')) is bool
                for v in checked.values()), 'Invalid local validation flags')
    passed = all(not v['new_material_error'] for v in checked.values())
    require(candidate.get('coverage') == OWNER_COUNT and candidate.get('local_gate_passed') is passed,
            'Local gate does not match all-owner validation')
    require(candidate.get('external_feedback_used') is False and candidate.get('human_reference_used') is False
            and candidate.get('source_fidelity_verified') is False, 'Invalid local evidence scope')
    require(candidate.get('status') in {'candidate_complete', 'unchanged_after_validation'}, 'Incomplete candidate')
    if round_id == 'Q3':
        require(source == rw.rows(folder / 'Q2/source.json'), 'Q3 cannot change its selected Japanese')
        previous = rw.read(folder / 'Q2/candidate.json')
        require(candidate['grounded_recap'] == previous['grounded_recap'], 'Q3 cannot regenerate grounded recaps')
    return candidate


def display_gate(folder: Path, round_id: str) -> dict:
    path = folder / round_id
    source, target = rw.rows(path / 'source.json'), rw.rows(path / 'target.json')
    out = path / 'display'
    document = rw.read(out / 'display.json')
    ds = parse_srt(out / 'source.srt', preserve_text_whitespace=True)
    dt = parse_srt(out / 'subtitles.zh.srt', preserve_text_whitespace=True)
    er.owner_rows_sha256(ds); er.owner_rows_sha256(dt)
    require([(r.index, r.ts_line) for r in ds] == [(r.index, r.ts_line) for r in dt], 'Display geometry differs')
    mapping = document.get('cues', [])
    require(document.get('semantic_text_changed') is False and len(mapping) == len(ds), 'Incomplete display receipt')
    lineage = [r.get('utterance') for r in mapping]
    require(all(type(i) is int for i in lineage) and lineage == sorted(lineage)
            and set(lineage) == set(range(1, OWNER_COUNT + 1)), 'Display owner lineage lost or reordered')
    for index, (s, t, row) in enumerate(zip(ds, dt, mapping), 1):
        require(row.get('line') == index and row.get('source') == s.text and row.get('target') == t.text,
                'Display receipt and rendered text differ')
    for s, t in zip(source, target):
        pieces = [row for row in mapping if row['utterance'] == s.index]
        require(''.join(x['source'] for x in pieces) == s.text and ''.join(x['target'] for x in pieces) == t.text,
                'Display changed owner text')
    return {'display_cues': len(dt), 'layout_warning_count': len(document.get('warnings', [])),
            'exact_owner_text': True, 'source_srt_sha256': file_hash(out / 'source.srt'),
            'target_srt_sha256': file_hash(out / 'subtitles.zh.srt')}


@contextmanager
def controller_lock(folder: Path):
    with (folder / 'loop.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Controller:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.conf = rw.check_inputs(self.folder)
        require(self.conf.get('maximum_candidates') == 3 and self.conf.get('maximum_local_seconds') == 5400
                and self.conf.get('maximum_round_seconds') == 1800, 'Campaign budget changed')
        self.raw = rw.rows(self.folder / 'source.json')
        owners_from_rows(self.raw)
        self.state_path = self.folder / 'loop-state.json'
        self.state = rw.read(self.state_path) if self.state_path.exists() else {
            'version': VERSION, 'status': 'prepared', 'created_utc': now(), 'rounds': {}, 'steps': {},
            'reviews': {}, 'selected_output_modified': False, 'eligible_candidate': None}
        require(self.state.get('version') == VERSION, 'Unknown controller state')

    def save(self):
        self.state['updated_utc'] = now()
        self.state['local_seconds'] = rw.read(self.folder / 'campaign.json')['local_seconds']
        write_json(self.state_path, self.state)

    def step(self, name, artifact: Path, operation, validate):
        """Adopt validated external caches; interrupted operations never repeat."""
        record = self.state['steps'].get(name)
        if record and record['status'] == 'failed':
            raise RuntimeError('A failed stage has already consumed its slot')
        if record and record['status'] == 'complete':
            require(artifact.is_file() and record['artifact_sha256'] == file_hash(artifact), 'Completed stage changed')
            return validate()
        if artifact.is_file():
            value = validate()
            self.state['steps'][name] = {'status': 'complete', 'adopted_cache': True,
                                        'artifact_sha256': file_hash(artifact), 'finished_utc': now()}
            self.save(); return value
        if record:
            raise RuntimeError('Interrupted stage cannot restart without a completed artifact')
        self.state['steps'][name] = {'status': 'running', 'started_utc': now()}
        self.save()
        try:
            operation()
            value = validate()
            self.state['steps'][name].update(status='complete', artifact_sha256=file_hash(artifact), finished_utc=now())
            self.save(); return value
        except BaseException as error:
            self.state['steps'][name].update(status='failed', error_type=type(error).__name__, finished_utc=now())
            self.save(); raise

    def diagnosis(self, round_id):
        artifact = self.folder / round_id / 'diagnosis.json'
        def validate():
            value = rw.read(artifact)
            source, target = rw.round_inputs(self.folder, round_id)
            require(value.get('source_hash') == fingerprint([asdict(x) for x in source])
                    and value.get('target_hash') == fingerprint([asdict(x) for x in target]), 'Diagnosis input drift')
            require(value.get('coverage') == OWNER_COUNT and set(value['audits']) == {str(i) for i in range(1,67)},
                    'Diagnosis coverage incomplete')
            select_deep(value)
            return value
        return self.step(round_id + ':diagnose', artifact, lambda: rw.diagnose(self.folder, round_id), validate)

    def evidence(self, round_id, mode, diagnosis=None):
        configured = self.conf.get('active_blind_evidence_dir') if mode == 'blind' else None
        path = Path(configured) if configured else self.folder / 'evidence' / mode
        if not path.is_absolute():
            path = (self.folder / path) if path.parts[0] == 'evidence' else ROOT / path
        path = path.resolve()
        require(path.is_relative_to(self.folder / 'evidence'), 'Evidence directory must remain in this campaign')
        ids, masking = (None, None) if mode == 'blind' else select_deep(diagnosis)
        if ids == []:
            return None
        def validate():
            value = audio.validate_evidence(path / 'evidence.json')
            require(any(o['status'] in {'ok', 'empty'} for o in value['observations']),
                    'No successful native observations; failed pool cannot support scoring')
            require(value['mode'] == mode and value['media_sha256'] == self.conf['media_sha256'], 'Wrong acoustic pool')
            expected = set(range(1, 67)) if mode == 'blind' else set(ids)
            expected_count = len(expected) * (2 if mode == 'blind' else 4)
            require(len(value['observations']) == expected_count
                    and value['maximum_asr_requests'] == expected_count
                    and {o['owner_id'] for o in value['observations']} == expected, 'Wrong scheduled acoustic coverage')
            # Validate the prepared crop policy too; cached results cannot substitute views.
            plan = rw.read(Path(value['plan_path']))
            prep = rw.read(Path(plan['preparation']['path']))
            request = prep['request']
            expected_owners = audio.validate_owners(owners_from_rows(self.raw), prep['master']['audio']['seconds'])
            require(request['owners'] == expected_owners and request['mode'] == mode
                    and request['flagged_ids'] == ids and request['masking_ids'] == masking
                    and request['audio_stream'] == 0, 'Cached acoustic owner/view policy changed')
            return value
        def acquire():
            require(not (path / 'execution-started.json').exists(), 'Existing acquisition cannot restart')
            with rw.measured(self.folder, round_id, mode + '_audio_revisit'):
                audio.acquire(Path(self.conf['media']), owners_from_rows(self.raw), path, mode,
                    flagged_ids=ids, masking_ids=masking, audio_stream=0,
                    bandit_manifest=Path(self.conf['bandit_manifest']) if mode == 'deep' and self.conf.get('bandit_manifest') else None,
                    python_executable=sys.executable, admin_url=rw.ADMIN, worker_timeout=1800, execute=True)
        self.step(round_id + ':' + mode, path / 'evidence.json', acquire, validate)
        return path / 'evidence.json'

    def bundle(self, label, source, target, evidence):
        path = self.folder / 'evaluation' / label / 'bundle' / 'manifest.json'
        if not path.exists():
            er.prepare_bundle(source, target, self.raw, evidence, self.folder / 'original-context.txt', path.parent)
        bundle = er.read_bundle(path)
        require(bundle['inputs']['source']['sha256'] == er.owner_rows_sha256(source)
                and bundle['inputs']['target']['sha256'] == er.owner_rows_sha256(target)
                and bundle['inputs']['raw']['sha256'] == er.owner_rows_sha256(self.raw)
                and bundle['inputs']['context']['sha256'] == self.conf['context_sha256'], 'Cached review bundle changed')
        expected = [audio.validate_evidence(p)['pool_version'] for p in evidence]
        actual = [x['pool_version'] for x in json.loads(bundle['raw_inputs']['evidence'])]
        require(actual == expected, 'Review evidence collection changed')
        return path

    def review(self, label, manifest, purpose, baseline=None, primary=None):
        path = self.folder / 'evaluation' / label / purpose
        kwargs = {'purpose': purpose, 'baseline_review': baseline, 'primary_review': primary}
        result = self.step('review:' + label + ':' + purpose, path / 'assessment.json',
                          lambda: er.review(manifest, path, execute=True, **kwargs),
                          lambda: er.validate_receipt(path, manifest))
        require(result['purpose'] == purpose, 'Cached review purpose changed')
        summary = {key: result[key] for key in ('score', 'pool_version', 'source_sha256', 'target_sha256',
                   'dispatch_id', 'started_utc', 'finished_utc', 'seconds', 'output_dir', 'receipt_hashes')}
        self.state['reviews'][label + ':' + purpose] = summary
        self.save()
        LOG.info('score %s %s = %s', label, purpose, result['score'])
        return result

    def baseline(self, pool_label, evidence):
        manifest = self.bundle(pool_label + '-baseline', self.raw, rw.rows(self.folder / 'target.json'), evidence)
        return self.review(pool_label + '-baseline', manifest, 'baseline')

    def score_candidate(self, round_id, pool_label, evidence, baseline):
        candidate = candidate_gate(self.folder, round_id)
        structural = display_gate(self.folder, round_id)
        source, target = rw.rows(self.folder / round_id / 'source.json'), rw.rows(self.folder / round_id / 'target.json')
        label = pool_label + '-' + round_id
        manifest = self.bundle(label, source, target, evidence)
        primary = self.review(label, manifest, 'candidate', baseline=baseline['output_dir'])
        record = self.state['rounds'].setdefault(round_id, {'status': 'candidate_complete'})
        pool_scores = record.setdefault('scores_by_pool', {}).setdefault(primary['pool_version'],
            {'pool_label': pool_label, 'confirmation_score': None})
        pool_scores.update(primary_score=primary['score'], primary_review=primary['output_dir'])
        record.update(score=primary['score'], score_pool_version=primary['pool_version'],
                      confirmation_score=pool_scores['confirmation_score'], status='primary_scored')
        self.save()
        if primary['score'] >= 4 and candidate['local_gate_passed'] and structural['exact_owner_text']:
            confirmed = self.review(label, manifest, 'confirmation', primary=primary['output_dir'])
            pool_scores.update(confirmation_score=confirmed['score'], confirmation_review=confirmed['output_dir'])
            record.update(confirmation_score=confirmed['score'], status='confirmation_scored')
            self.save()
            if confirmed['score'] >= 4:
                self.state['eligible_candidate'] = {'round': round_id, 'pool_version': primary['pool_version'],
                    'source_sha256': primary['source_sha256'], 'target_sha256': primary['target_sha256'],
                    'primary': primary['output_dir'], 'confirmation': confirmed['output_dir'],
                    'minimum_score': min(primary['score'], confirmed['score']), 'structural': structural,
                    'local_gate_scope': candidate['local_gate_scope'], 'promoted': False}
                record['status'] = 'confirmed_candidate'
                self.state['status'] = 'confirmed_candidate'; self.save(); return True
        return False

    def duplicate_target(self, round_id):
        target = er.owner_rows_sha256(rw.rows(self.folder / round_id / 'target.json'))
        previous = [('baseline', self.folder / 'target.json')]
        previous += [(r, self.folder / r / 'target.json') for r in ROUND_IDS[:ROUND_IDS.index(round_id)]]
        return next((label for label, path in previous if path.exists()
                     and er.owner_rows_sha256(rw.rows(path)) == target), None)

    def local_candidate(self, round_id, evidence):
        path = self.folder / round_id
        candidate = self.step(round_id + ':repair', path / 'candidate.json',
            lambda: rw.repair(self.folder, round_id, evidence), lambda: candidate_gate(self.folder, round_id))
        if candidate['status'] == 'unchanged_no_actionable_questions':
            self.state['status'] = 'no_actionable_questions'; self.save(); return False
        duplicate = self.duplicate_target(round_id)
        if duplicate is not None:
            self.state['rounds'][round_id].update(status='unchanged_target', same_wording_as=duplicate, score=None)
            self.state['status'] = 'unchanged_target'; self.save(); return False
        self.step(round_id + ':display', path / 'display/display.json', lambda: rw.display(self.folder, round_id),
                  lambda: display_gate(self.folder, round_id))
        self.state['rounds'][round_id].update(status='candidate_complete', local_gate_passed=candidate['local_gate_passed'],
            changed_owner_count=len(candidate['changed_owners']), changed_source_owner_count=len(candidate['changed_source_owners']))
        self.save(); return True

    def reserve_round(self, round_id):
        require(round_id in ROUND_IDS, 'Fourth candidate is not authorized')
        prior = self.state['rounds'].get(round_id)
        require(prior is None or prior.get('status') != 'failed', 'Failed candidate slot cannot restart')
        require(rw.remaining(self.folder, round_id) > 0, 'Local campaign or round budget exhausted')
        if prior is None:
            self.state['rounds'][round_id] = {'status': 'reserved', 'reserved_utc': now(), 'score': None}
            self.save()

    def run_rounds(self):
        evidence = []
        for round_id in ROUND_IDS:
            self.reserve_round(round_id)
            self.state['active_round'] = round_id; self.save()
            diagnosis = self.diagnosis(round_id)
            if round_id == 'Q1':
                evidence = [self.evidence(round_id, 'blind')]
                pool_label = 'blind'
                baseline = self.baseline(pool_label, evidence)
            elif round_id == 'Q2':
                deep = self.evidence(round_id, 'deep', diagnosis)
                if deep is not None:
                    evidence.append(deep); pool_label = 'deep'
                    baseline = self.baseline(pool_label, evidence)
                    if self.score_candidate('Q1', pool_label, evidence, baseline):
                        return
                # No audio need means target-only diagnosed errors may still be actionable.
            if not diagnosis['questions'] and round_id != 'Q1':
                self.state['status'] = 'no_actionable_questions'; self.save(); return
            if not self.local_candidate(round_id, evidence):
                return
            if self.score_candidate(round_id, pool_label, evidence, baseline):
                return
            key = pool_label + '-' + round_id + ':candidate'
            self.state['rounds'][round_id]['score'] = self.state['reviews'][key]['score']
            self.state['rounds'][round_id]['status'] = 'scored_below_release'
            self.save()
        self.state['status'] = 'candidate_budget_exhausted'; self.save()

    def execute(self):
        with controller_lock(self.folder):
            if self.state['status'] in {'confirmed_candidate', 'no_actionable_questions', 'unchanged_target',
                                       'candidate_budget_exhausted', 'failed'}:
                return self.state
            files = [Path(__file__), Path(rw.__file__), Path(audio.__file__), Path(er.__file__),
                     ROOT / 'src/contextual_review.py', ROOT / 'src/contextual_review_contract.py']
            pins = {str(p.resolve()): file_hash(p) for p in files}
            if 'code_pins' in self.state:
                require(self.state['code_pins'] == pins, 'Controller or frozen stage code changed')
            else:
                self.state['code_pins'] = pins
            require(file_hash(Path(self.conf['media'])) == self.conf['media_sha256'], 'Campaign media changed')
            initial = rw._request_json(rw.ADMIN + '/status', admin=True)
            model = initial.get('loaded_model')
            require(isinstance(model, str) and bool(model) or model is None and initial.get('state') == 'idle',
                    'Cannot capture original backend state')
            self.state.update(status='running', initial_backend={'loaded_model': model, 'state': initial.get('state')},
                              original_backend_restored=False)
            self.save()
            try:
                self.run_rounds()
            except BaseException as error:
                round_id = self.state.get('active_round')
                if round_id in self.state['rounds']:
                    self.state['rounds'][round_id].update(status='failed', error_type=type(error).__name__)
                self.state.update(status='failed', error_type=type(error).__name__)
                raise
            finally:
                try:
                    after = rw._request_json(rw.ADMIN + '/status', admin=True)
                    if after.get('loaded_model') != model or model is None and after.get('state') != 'idle':
                        if model is None:
                            rw._request_json(rw.ADMIN + '/unload', {}, admin=True, timeout=45)
                        else:
                            rw._request_json(rw.ADMIN + '/load', {'model': model}, admin=True, timeout=360)
                    after = rw._request_json(rw.ADMIN + '/status', admin=True)
                    require(after.get('loaded_model') == model and (model is not None or after.get('state') == 'idle'),
                            'Original backend restoration failed')
                    self.state['original_backend_restored'] = True
                except BaseException as error:
                    self.state.update(status='failed', restoration_error_type=type(error).__name__, eligible_candidate=None)
                    raise
                finally:
                    self.save()
            return self.state


def public_status(state):
    return {key: state.get(key) for key in ('version', 'status', 'active_round', 'local_seconds',
            'eligible_candidate', 'selected_output_modified', 'original_backend_restored', 'error_type')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'output/quality-revisit-20260914')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        controller = Controller(args.campaign)
        state = controller.execute() if args.execute else controller.state
        print(json.dumps(public_status(state), ensure_ascii=False))
        return 1 if state['status'] == 'failed' else 0
    except BaseException as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'executed': args.execute}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
