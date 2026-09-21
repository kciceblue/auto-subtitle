"""Two fixed, primary-only full-draft scores; never a qualification or writer.

--prepare performs CPU replay and freezes both bundles. Only --execute can
invoke the existing score-only reviewer. No campaign producer is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import contextual_evidence_review as er
from src import episode_draft_requests as lrequests
from src import episode_draft_recovery as recovery
from src import episode_draft_release as lrelease
from src import episode_draft_revisit as lcontroller
from src import readable_draft_revisit as rcontroller
from src import readable_draft_release as rrelease
from src import readable_draft_native as rnative
from src import whole_owner as wo
from src import whole_owner_requests as owner_requests
from src import joint_context_release as exact_native
from src import temporal_release as temporal
from src import sparse_release as provenance
from src import sparse_revisit as sr
from src import revisit_workflow as rw
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'full-draft-primary-diagnostic-1'
FOLDER = ROOT/'output/quality-full-draft-diagnostic-20260915'
PLAN = ROOT/'design/quality-full-draft-diagnostic-20260915.md'
AUDIT = ROOT/'output/quality-episode-draft-20260915/diagnostics/full-draft-counterfactual'
SCORE_INDEX = AUDIT/'score-index.json'
ADDITIONAL_INDEX = AUDIT/'additional-score-indexes.json'
PARENTS = {'DF-L1': ROOT/'output/quality-episode-draft-20260915',
           'DF-R1': ROOT/'output/quality-readable-draft-20260915'}
TARGETS = {'DF-L1': '938f1c9367fa38a0cc32e9fbf20698f5c9c61f1b3b37fdf4298a8d1eb9914d03',
           'DF-R1': '231c57d9d6f61e64d9ad80127d3e217672859b4bf1f2abfbdf124f6c4333b8d3'}
POOL = 'cf71e2d12a6905c8ca728b00a9a548567b4f90fecd63e9d988841ead96c31a13'
SOURCE = '807e6b35f2099421f68ef326813123d8f4a9c958afe179a0e68540641dbf33a1'
CONTEXT = '7ea692a93e6c2a9bc64bab0d7e8c58ddc47b07a268055de25c11d4c85f3bbf21'
STABLE_KEYS = ('version', 'input_hashes', 'code_pins', 'producer_snapshot_manifest',
    'source_sha256', 'target_sha256', 'context_sha256', 'pool_version', 'evidence_paths',
    'baseline_review', 'source_pack_sha256', 'map_sha256', 'identity_map_sha256',
    'readable_projection_sha256', 'writer_recipe', 'writer_recipe_sha256', 'media',
    'media_sha256', 'predecessor', 'l1_predecessor', 'joint_predecessor')
LOG = logging.getLogger(__name__)


def read(path: Path):
    return strict_json(Path(path).read_bytes())


def same(a, b):
    return fingerprint(a) == fingerprint(b)


def stable(conf):
    return {key: conf[key] for key in STABLE_KEYS if key in conf}


def immutable(path: Path, value, *, raw=False):
    data = value if raw else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
    path.chmod(0o444)


@contextmanager
def locked(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Inputs:
    """Pin static artifacts; take a newline-complete prefix of live R metrics."""
    def __init__(self, folder):
        self.folder = folder
        self.pins = {}
        self.live = PARENTS['DF-R1']/'R1/requests/metrics.jsonl'
        self.campaign = PARENTS['DF-R1']/'campaign.json'
        self.prefix = None
        self.stages = {}

    def __call__(self, path, raw=False):
        path = Path(path).resolve()
        data = path.read_bytes()
        if path == self.live:
            if self.prefix is None:
                self.prefix = data[:data.rfind(b'\n')+1]
                require(bool(self.prefix), 'Live writer telemetry is incomplete')
            require(data.startswith(self.prefix), 'Live telemetry prefix changed')
            data = self.prefix
        elif path != self.campaign:
            digest = hashlib.sha256(data).hexdigest()
            require(str(path) not in self.pins or self.pins[str(path)] == digest,
                    'Static native input changed during preparation')
            self.pins[str(path)] = digest
        return data if raw else strict_json(data)

    def seal_stage(self, key):
        rows = [strict_json(line) for line in self.prefix.splitlines()]
        self.stages[key] = [row for row in rows if row.get('stage') == key]
        require(bool(self.stages[key]), 'Sealed native stage has no telemetry')

    def finish(self):
        path = self.folder/'native-inputs/R1-metrics-prefix.jsonl'
        immutable(path, self.prefix, raw=True)
        self.pins[str(path)] = file_hash(path)
        return {'path': str(self.live), 'snapshot': str(path), 'bytes': len(self.prefix),
                'sha256': hashlib.sha256(self.prefix).hexdigest(), 'sealed_stages': self.stages}


def replay_rejection(folder, source, target, proposals, pack, bound, track):
    for tx in proposals:
        path = folder/'R1/requests'/('verify-'+tx['transaction_id']+'.json')
        if not path.exists():
            continue
        candidate = read(path)
        if candidate.get('status') != 'complete':
            continue  # An active verifier is neither a result nor a static pin.
        expected = [row for attempt in candidate.get('attempts', [])
                    for row in attempt.get('native_receipts', [])]
        captured = [strict_json(line) for line in track.prefix.splitlines()]
        captured = [row for row in captured if row.get('stage') == path.stem]
        if len(captured) < len(expected) and same(captured, expected[:len(captured)]):
            continue  # Completed after the sealed telemetry snapshot; leave it unpinned.
        require(same(captured, expected), 'R rejection stage has unregistered native telemetry')
        request = owner_requests.build_verifier_request(pack, bound, source, target, tx)
        exact_native._assert_native_request(path, request, 'qwen', track)
        verdict = temporal._native(path, request['instruction'], request['body'], request['schema'], 'qwen', track)
        if not wo.accept_owner_decision(verdict, tx, pack):
            require(tx['valid'] is True and tx['no_op'] is False, 'Rejected transaction is not a distinct rewrite')
            track.seal_stage(path.stem)
            return {'transaction_id': tx['transaction_id'], 'transaction_sha256': tx['transaction_sha256'],
                    'request_path': str(path), 'request_sha256': file_hash(path),
                    'verdict_sha256': fingerprint(verdict), 'accepted': False}
    raise ValueError('A completed strict non-noop R rejection is required before scoring')


def reconstruct(folder, track):
    left, right = PARENTS.values()
    lc, rc = lcontroller.check(left), rcontroller.check(right)
    require(lc.get('original_backend_restored') is True and lc.get('status') not in {'prepared', 'running'},
            'L1 must be terminal and restored')
    require(lc.get('native_attempt_recovery'), 'Fixed L draft requires registered mechanical recovery')
    for parent, conf in ((left, lc), (right, rc)):
        provenance._producer_snapshots(parent, conf, track)
        for path, digest in {**conf['input_hashes'], **conf['code_pins']}.items():
            track(path, raw=True)
            require(file_hash(Path(path)) == digest, 'Producer or source pin changed')
    track(left/'campaign.json')
    source, target = rw.rows(left/'source.json'), rw.rows(left/'target.json')
    pools = [er._validate_pool(Path(path)) for path in lc['evidence_paths']]
    pack, bound, baseline, _ = lrelease._source_ancestry(left, lc, source, target, pools, track)
    for key in ('source_sha256', 'target_sha256', 'context_sha256', 'pool_version',
                'evidence_paths', 'baseline_review', 'source_pack_sha256', 'map_sha256'):
        require(same(lc[key], rc[key]), 'L and R immutable source/pool/context differ')
    for name in ('source.json', 'target.json', 'original-context.txt', 'source-pack.json', 'source-map.json'):
        require(track(left/name, raw=True) == track(right/name, raw=True), 'L and R source artifacts differ')
    require(baseline['purpose'] == 'baseline' and baseline['score'] == 2 and baseline['pool_version'] == POOL
            and er.owner_rows_sha256(source) == SOURCE and lc['context_sha256'] == CONTEXT,
            'Declared same-pool baseline or original inputs differ')
    requests = {'DF-L1': lrequests.build_writer_request(pack, bound)}
    identities, projection = rcontroller.prepared_readable(right, pack, bound)
    requests['DF-R1'] = rcontroller.writer_request(pack, bound, identities, projection)
    independently_built, _, _ = rrelease._readable_inputs(right, rc, pack, bound, track)
    require(same(requests['DF-R1'], independently_built), 'Readable request differs from native identity reconstruction')
    raws = {'DF-L1': recovery.replay_native(left, requests['DF-L1'], track),
            'DF-R1': rnative.replay_native(right/'R1/requests/readable-draft.json', requests['DF-R1'], track)}
    track.seal_stage('readable-draft')
    arms = []; prepared = []
    for label, parent in PARENTS.items():
        round_id = label[3:]
        raw = raws[label]
        require(set(raw) == {'owners'} and set(raw['owners']) == set(map(str, range(1, 67))), 'Draft coverage differs')
        proposals, draft = compile_draft(label, raw, source, target, pack,
                                         track(parent/round_id/'proposed-transactions.json'))
        if label == 'DF-R1':
            rejection = replay_rejection(parent, source, target, proposals, pack, bound, track)
        prepared.append((label, parent, proposals, draft))
    # Native/canonical proof for BOTH arms, including the existing R rejection,
    # precedes any bundle or proposal write.
    for label, parent, proposals, draft in prepared:
        bundle = er.prepare_bundle(source, draft, source, list(map(Path, lc['evidence_paths'])),
                                   left/'original-context.txt', folder/label/'bundle')
        immutable(folder/label/'proposed-transactions.json', proposals)
        arms.append({'label': label, 'parent': str(parent), 'target_sha256': TARGETS[label],
            'source_sha256': SOURCE, 'pool_version': POOL, 'context_sha256': CONTEXT,
            'writer_request_sha256': fingerprint(requests[label]), 'manifest': str(bundle),
            'baseline_review': baseline['output_dir'], 'primary_directory': str(folder/label/'primary'),
            'proposals_sha256': fingerprint(proposals), 'owners': 66, 'normalization_changes': 0})
    return arms, stable(rc), rejection


def compile_draft(label, raw, source, target, pack, saved):
    proposals = [wo.prepare_owner(owner, raw['owners'][str(owner)]['chinese'], source, target, pack)
                 for owner in range(1, 67)]
    require(all(tx['valid'] is True and tx['no_op'] is False and tx['target_normalized'] is False
                for tx in proposals), 'Full draft has invalid/noop/normalized proposals')
    require(same(proposals, saved), 'Saved proposals differ from native draft')
    for tx in proposals:
        wo.validate_owner_transaction(tx, pack, source=source, target=target)
    new_source, draft, _ = wo.apply_owner_transactions(source, target, proposals, pack)
    require(same([asdict(row) for row in new_source], [asdict(row) for row in source])
            and er.owner_rows_sha256(draft) == TARGETS[label], 'Full draft/source hash differs from declaration')
    return proposals, draft


def load_score_index(track):
    index, additional = track(SCORE_INDEX), track(ADDITIONAL_INDEX)
    require(not index.get('load_errors') and not additional.get('load_errors'), 'Score catalog contains load errors')
    for row in index['reviews']:
        require(not row.get('metadata_hash_conflicts'), 'Score metadata has conflicting hashes')
    records = list(index['records'])
    for row in additional['additional_score_wrapper_records']:
        records.append({**row, 'hashes': {key: row.get(key) for key in
            ('pool_version', 'target_sha256', 'source_sha256')}})
    for row in records:
        track(row['path'], raw=True)
        require(file_hash(Path(row['path'])) == row['metadata_sha256'], 'Historical score catalog metadata changed')
    return {**index, 'records': records, 'additional_index_sha256': file_hash(ADDITIONAL_INDEX)}


def matches(metadata, arm):
    inputs = metadata.get('inputs', {})
    return (metadata.get('pool_version') == arm['pool_version']
        and inputs.get('target', {}).get('sha256') == arm['target_sha256'])


def duplicates(index, arm, folder):
    """Recheck named historical receipts plus new output review ledgers; metadata only."""
    require(not index.get('load_errors'), 'Duplicate-score index has load errors')
    directories = {Path(row['directory']) for row in index['reviews']}
    directories.update(path.parent for path in (ROOT/'output').rglob('review-run.json'))
    directories.update(path.parent for path in (ROOT/'output').rglob('dispatch-reservation.json'))
    completed = []
    for directory in sorted(directories):
        if directory == Path(arm['primary_directory']):
            continue
        path = directory/'review-run.json'
        if not path.exists():
            preparation = directory/'preparation.json'
            if preparation.exists() and matches(read(preparation), arm):
                raise ValueError('Matching prior review was reserved without completion; no reroll')
            continue
        meta = read(path)
        if not matches(meta, arm):
            continue
        require(meta.get('protocol') == er.VERSION and meta.get('status') == 'completed',
                'Matching prior review failed or remains incomplete; no reroll')
        receipt = er.validate_receipt(directory)
        validate_score(receipt, arm)
        if receipt['purpose'] == 'confirmation':
            receipt = er.validate_receipt(receipt['dependency']['output_dir'])
            validate_score(receipt, arm)
        require(receipt['purpose'] in {'candidate', 'baseline'}, 'Exact duplicate has no primary receipt')
        completed.append(receipt)
    # A wrapper claiming this target without an enumerated native review blocks dispatch.
    for record in index.get('records', []):
        hashes = record.get('hashes', {})
        if hashes.get('pool_version') == POOL and hashes.get('target_sha256') == arm['target_sha256']:
            require(bool(completed), 'Matching score index lacks a validated native primary')
    if not completed:
        return None
    unique = {row['dispatch_id']: row for row in completed}
    # Preserve the earliest exact primary rather than selecting its most favorable reroll.
    return min(unique.values(), key=lambda row: (row['started_utc'], row['dispatch_id']))


def validate_score(receipt, arm):
    require(receipt['pool_version'] == arm['pool_version']
        and receipt['inputs']['source']['sha256'] == arm['source_sha256']
        and receipt['inputs']['raw']['sha256'] == arm['source_sha256']
        and receipt['inputs']['context']['sha256'] == arm['context_sha256']
        and receipt['inputs']['target']['sha256'] == arm['target_sha256'], 'Score is not for these exact inputs')


def validate_registration(folder):
    reg = read(folder/'registration.json')
    require(reg['registration_sha256'] == fingerprint({k:v for k,v in reg.items() if k != 'registration_sha256'})
        and reg['version'] == VERSION and reg['diagnostic_only'] is True and reg['qualified'] is False
        and reg['maximum_primary_dispatches'] == 2 and reg['maximum_confirmations'] == 0
        and [arm['label'] for arm in reg['arms']] == list(TARGETS), 'Diagnostic registration changed')
    for path, digest in reg['pins'].items():
        require(file_hash(Path(path)) == digest, 'Diagnostic input/script/bundle pin changed')
    require(reg['pins'].get(str(PLAN)) == file_hash(PLAN)
        and reg['pins'].get(str(Path(__file__).resolve())) == file_hash(Path(__file__)), 'Diagnostic plan/script missing')
    rc = rcontroller.check(PARENTS['DF-R1'])
    require(same(stable(rc), reg['r_stable_campaign']), 'Active R stable input or producer identity changed')
    prefix = reg['r_metrics_prefix']; data = Path(prefix['path']).read_bytes()
    saved = Path(prefix['snapshot']).read_bytes()
    require(len(saved) == prefix['bytes'] and hashlib.sha256(saved).hexdigest() == prefix['sha256']
        and data.startswith(saved), 'Preserved R writer/checker telemetry prefix changed')
    complete = data[:data.rfind(b'\n')+1]
    rows = [strict_json(line) for line in complete.splitlines()]
    for stage, expected in prefix['sealed_stages'].items():
        require(same([row for row in rows if row.get('stage') == stage], expected), 'Sealed R native stage reran')
    require(reg['r_strict_rejection']['accepted'] is False, 'Missing strict R rejection')
    for arm in reg['arms']:
        require(arm['target_sha256'] == TARGETS[arm['label']] and arm['pool_version'] == POOL
            and arm['source_sha256'] == SOURCE and arm['context_sha256'] == CONTEXT, 'Declared diagnostic input changed')
        bundle = er.read_bundle(arm['manifest']); validate_score(bundle, arm)
    return reg


def prepare(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        if (folder/'registration.json').exists():
            return validate_registration(folder)
        require(set(folder.iterdir()) == {folder/'.lock'}, 'Unregistered diagnostic artifacts already exist')
        started = time.monotonic(); track = Inputs(folder)
        track(PLAN, raw=True); track(Path(__file__), raw=True)
        index = load_score_index(track)
        immutable(folder/'score-index-before.json', index)
        arms, r_stable, rejection = reconstruct(folder, track)
        prefix = track.finish()
        for arm in arms:
            previous = duplicates(index, arm, folder)
            arm['previous_primary'] = previous['output_dir'] if previous else None
            if previous:
                for name, digest in previous['receipt_hashes'].items():
                    path = Path(previous['output_dir'])/name
                    track(path, raw=True)
                    require(file_hash(path) == digest, 'Existing exact primary changed')
        for path in folder.rglob('*'):
            if path.is_file() and path.name != '.lock':
                track.pins[str(path)] = file_hash(path)
        reg = {'version': VERSION, 'created_utc': sr.now(), 'diagnostic_only': True, 'qualified': False,
            'maximum_primary_dispatches': 2, 'maximum_confirmations': 0, 'new_local_inference': False,
            'new_wording': False, 'human_reference_sent': False, 'audio_sent': False, 'video_sent': False,
            'local_negatives_unchanged': True, 'cpu_prepare_to_seal_seconds': time.monotonic()-started,
            'arms': arms, 'r_stable_campaign': r_stable, 'r_metrics_prefix': prefix,
            'r_strict_rejection': rejection, 'pins': track.pins}
        reg['registration_sha256'] = fingerprint(reg)
        immutable(folder/'registration.json', reg)
        write_json(folder/'diagnostic.json', {'version': VERSION, 'diagnostic_only': True, 'qualified': False,
            'confirmed': False, 'status': 'prepared', 'registration_sha256': reg['registration_sha256'], 'scores': []})
        checked = validate_registration(folder)
        state = read(folder/'diagnostic.json')
        state['cpu_prepare_seconds'] = time.monotonic()-started
        write_json(folder/'diagnostic.json', state)
        return checked


def execute(folder=FOLDER):
    folder = Path(folder).resolve()
    with locked(folder):
        reg = validate_registration(folder)
        state = read(folder/'diagnostic.json')
        require(state['registration_sha256'] == reg['registration_sha256']
            and state['status'] != 'failed' and state['diagnostic_only'] is True
            and state['qualified'] is False and state['confirmed'] is False, 'Terminal or altered diagnostic cannot resume')
        index = read(folder/'score-index-before.json')
        scores = state['scores']
        require([row['label'] for row in scores] == list(TARGETS)[:len(scores)] and len(scores) <= 2,
                'Primary order or count changed')
        for position, arm in enumerate(reg['arms']):
            validate_registration(folder)  # Both frozen bundles before every possible dispatch.
            if position < len(scores):
                receipt = er.validate_receipt(scores[position]['primary'])
                validate_score(receipt, arm)
                require(receipt['purpose'] in {'candidate', 'baseline'}, 'Cached score is not a primary')
                require(receipt['score'] == scores[position]['score'], 'Cached scalar changed')
                continue
            primary = Path(arm['primary_directory'])
            try:
                previous = duplicates(index, arm, folder)
                if arm['previous_primary']:
                    pinned = er.validate_receipt(arm['previous_primary']); validate_score(pinned, arm)
                    previous = pinned
                started = time.monotonic()
                if (primary/'assessment.json').exists():
                    receipt = er.validate_receipt(primary, arm['manifest'])
                    reused = True
                elif previous:
                    require(not state.get('active_label') and not any(primary.glob('dispatch-*'))
                        and not any((primary/name).exists() for name in ('review-run.json', 'review-process.log', 'review-response.json')),
                        'Interrupted or failed primary cannot be bypassed by another receipt')
                    receipt = previous; reused = True
                else:
                    require(not state.get('active_label') and not any(primary.glob('dispatch-*'))
                        and not any((primary/name).exists() for name in ('review-run.json', 'review-process.log', 'review-response.json')),
                        'Interrupted or failed primary cannot reroll')
                    state.update(status='running', active_label=arm['label'], dispatch_reserved_utc=sr.now())
                    write_json(folder/'diagnostic.json', state)
                    receipt = sr.review_once(arm['manifest'], primary, purpose='candidate',
                                             baseline_review=arm['baseline_review'])
                    reused = False
                validate_score(receipt, arm)
                receipt = er.validate_receipt(receipt['output_dir'])
                require(receipt['purpose'] in {'candidate', 'baseline'}, 'Diagnostic score is not a primary')
                scores.append({'label': arm['label'], 'status': 'complete', 'score': receipt['score'],
                    'primary': receipt['output_dir'], 'dispatch_id': receipt['dispatch_id'],
                    'receipt_hashes': receipt['receipt_hashes'], 'target_sha256': arm['target_sha256'],
                    'source_sha256': SOURCE, 'pool_version': POOL, 'context_sha256': CONTEXT,
                    'reused_completed_primary': reused, 'review_seconds': receipt['seconds'],
                    'runner_score_seconds': time.monotonic()-started, 'diagnostic_only': True,
                    'qualified': False, 'confirmed': False})
                state.pop('active_label', None)
                state.update(status='complete' if len(scores) == 2 else 'running', scores=scores)
                write_json(folder/'diagnostic.json', state)
                # Small discoverable index for future exact-target score inheritance.
                write_json(folder/arm['label']/'score.json', scores[-1])
            except BaseException as exc:
                state.update(status='failed', failed_label=arm['label'], error_type=type(exc).__name__, finished_utc=sr.now())
                write_json(folder/'diagnostic.json', state)
                raise
        state.update(status='complete', finished_utc=sr.now())
        write_json(folder/'diagnostic.json', state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        result = prepare() if args.prepare else execute()
        LOG.info('diagnostic_only=true qualified=false status=%s completed_primaries=%d',
                 result.get('status', 'prepared'), len(result.get('scores', [])))
        return 0
    except Exception as exc:
        LOG.error('Diagnostic stopped: %s (details retained privately)', type(exc).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
