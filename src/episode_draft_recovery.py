"""One mechanical replay of the zero-based native telemetry validation failure.

No network or generation API is used here. The original failed native ledger is
never rewritten; a corrected completion view exists only in memory for replay.
"""
from __future__ import annotations
from copy import deepcopy
import argparse
from pathlib import Path
import time
from src import episode_draft_native as native
from src.contextual_review_contract import require, strict_json
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'episode-native-attempt-recovery-1'
DIRECTORY = 'recovery-native-attempt'
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'design/quality-episode-draft-recovery-20260915.md'


def _track(path, raw=False):
    data = Path(path).read_bytes()
    return data if raw else strict_json(data)


def _derive(saved):
    require(isinstance(saved, dict) and set(saved) == native._ROOT_KEYS
        and saved['receipt_sha256'] == fingerprint({k:v for k,v in saved.items() if k != 'receipt_sha256'})
        and saved['status'] == 'failed' and saved['parsed'] is None
        and len(saved['attempts']) == 1, 'Recovery requires the original sole failed ledger')
    attempt = saved['attempts'][0]
    require(attempt['number'] == 1 and attempt['error_type'] == 'ValueError'
        and attempt['failure_kind'] == 'provenance'
        and len(attempt['native_receipts']) == 1
        and type(attempt['native_receipts'][0]['attempt']) is int
        and attempt['native_receipts'][0]['attempt'] == 0,
        'Recovery failure is not the declared telemetry index defect')
    native._receipts(attempt, saved['request'], 'episode-draft')
    parsed = native._parse_complete(attempt, saved['request']['schema'])
    result = deepcopy(saved)
    result.update(status='complete', parsed=parsed)
    result['attempts'][0].update(error_type=None, failure_kind=None)
    result['receipt_sha256'] = fingerprint({k:v for k,v in result.items() if k != 'receipt_sha256'})
    return result


def _historical_epoch(before):
    return {'name':'L1-before-native-attempt-recovery', 'code_pins':before['code_pins'],
        'producer_snapshot_manifest':before['producer_snapshot_manifest'],
        'producer_snapshot_manifest_sha256':file_hash(Path(before['producer_snapshot_manifest'])),
        'reason':'Correct only zero-based native telemetry validation; reuse original complete draft',
        'ended_utc':before['finished_utc']}


def validate_registration(folder, conf, track=None):
    folder = Path(folder).resolve(); track = track or _track
    path = folder/DIRECTORY/'recovery.json'
    require(conf.get('native_attempt_recovery') == str(path)
        and conf['input_hashes'].get(str(path)) == file_hash(path), 'Unregistered native attempt recovery')
    rec = track(path)
    require(rec['version'] == VERSION and rec['maximum_new_writer_requests'] == 0
        and rec['maximum_new_asr_requests'] == 0, 'Recovery changed semantic attempt limits')
    for name,digest in rec['artifacts'].items():
        track(Path(name), raw=True)
        require(file_hash(Path(name)) == digest and conf['input_hashes'].get(name) == digest,
            'Original recovery artifact changed or unpinned')
    before = track(Path(rec['campaign_before']))
    require(before['status'] == 'failed' and before['error_type'] == 'ValueError'
        and before['original_backend_restored'] is True
        and before['local_seconds'] == rec['spent_local_seconds']
        and conf['local_seconds'] >= before['local_seconds']
        and conf['maximum_candidates'] == before['maximum_candidates'] == 1
        and conf['maximum_local_seconds'] == before['maximum_local_seconds'] == 7200
        and conf['maximum_round_seconds'] == before['maximum_round_seconds'] == 7200,
        'Recovery erased failure history or reset its budget')
    old_stages = before['rounds']['L1']['stages']
    require(conf['rounds']['L1']['stages'][:len(old_stages)] == old_stages
        and conf['rounds']['L1']['local_seconds'] >= before['rounds']['L1']['local_seconds'],
        'Recovery erased original stage accounting')
    require(conf.get('code_epochs') == [_historical_epoch(before)], 'Recovery producer epoch changed')
    old_manifest = track(Path(before['producer_snapshot_manifest']))
    mandatory = {str(PLAN), str(folder/DIRECTORY/'campaign-before.json'),
        str(folder/DIRECTORY/'native-metrics-before.jsonl'), str(folder/'L1/requests/episode-draft.json'),
        before['producer_snapshot_manifest'], *(spec['snapshot'] for spec in old_manifest.values())}
    require(mandatory <= set(rec['artifacts'])
        and rec['campaign_before'] == str(folder/DIRECTORY/'campaign-before.json')
        and rec['native_metrics_before'] == str(folder/DIRECTORY/'native-metrics-before.jsonl')
        and all(conf['input_hashes'].get(k) == v for k,v in before['input_hashes'].items()),
        'Recovery omitted mandatory original provenance or input pins')
    require(set(old_manifest) == set(before['code_pins']), 'Original producer coverage changed')
    for original,spec in old_manifest.items():
        data = track(Path(spec['snapshot']), raw=True)
        require(file_hash(Path(spec['snapshot'])) == spec['sha256'] == before['code_pins'][original],
            'Original producer snapshot changed')
    original_native = track(Path(old_manifest[str(ROOT/'src/episode_draft_native.py')]['snapshot']), raw=True)
    corrected_native = track(ROOT/'src/episode_draft_native.py', raw=True)
    expected = original_native.replace(b"receipt['attempt'] == 1", b"receipt['attempt'] == 0")
    require(original_native.count(b"receipt['attempt'] == 1") == 1 and corrected_native == expected,
        'Recovery altered native behavior beyond the inner attempt index')
    return rec


def replay_native(folder, request, track):
    folder = Path(folder).resolve()
    conf = track(folder/'campaign.json')
    rec = validate_registration(folder, conf, track)
    path = folder/'L1/requests/episode-draft.json'
    saved = track(path)
    original_metrics = track(Path(rec['native_metrics_before']), raw=True)
    metrics = track(path.parent/'metrics.jsonl', raw=True)
    require(metrics.startswith(original_metrics), 'Original native telemetry prefix changed')
    original_rows = native._metrics(original_metrics)
    require(len(original_rows) == 1 and original_rows == saved['attempts'][0]['native_receipts'],
        'Recovery has hidden pre-failure generation')
    derived = _derive(saved)
    require(derived['receipt_sha256'] == rec['derived_receipt_sha256']
        and fingerprint(derived['parsed']) == rec['parsed_sha256'], 'Mechanical derivation changed')
    def virtual(path_to_read, raw=False):
        if Path(path_to_read).resolve() == path:
            require(not raw, 'Unexpected raw virtual ledger access')
            return deepcopy(derived)
        return track(path_to_read, raw=raw)
    return native.replay_native(path, request, virtual)


def register(folder):
    from src import episode_draft_revisit as controller
    from src import sparse_revisit as sr
    start = time.monotonic(); folder = Path(folder).resolve()
    before = _track(folder/'campaign.json')
    require(before['version'] == controller.VERSION and before['status'] == 'failed'
        and before.get('error_type') == 'ValueError' and before.get('original_backend_restored') is True
        and not before.get('native_attempt_recovery'), 'Campaign is not eligible for one mechanical recovery')
    out = folder/'L1'; directory = folder/DIRECTORY
    require(not directory.exists() and not any((out/name).exists() for name in
        ('proposed-transactions.json','local-decisions.json','candidate.json','score.json','display')),
        'Recovery already exists or downstream work began')
    for name,digest in before['input_hashes'].items():
        require(file_hash(Path(name)) == digest, 'Original input changed before recovery')
    paths = list((out/'requests').glob('*.json'))
    require(paths == [out/'requests/episode-draft.json'], 'Unexpected native request before recovery')
    saved = _track(paths[0]); derived = _derive(saved)
    metrics = (out/'requests/metrics.jsonl').read_bytes()
    require(native._metrics(metrics) == saved['attempts'][0]['native_receipts'], 'Unexpected pre-recovery calls')
    request = controller.writer_request(_track(folder/'source-pack.json'), _track(folder/'source-map.json'))
    def virtual(path, raw=False):
        return deepcopy(derived) if Path(path) == paths[0] else _track(path,raw)
    native.replay_native(paths[0], request, virtual)
    directory.mkdir()
    (directory/'campaign-before.json').write_bytes((folder/'campaign.json').read_bytes())
    (directory/'native-metrics-before.jsonl').write_bytes(metrics)
    artifacts = {str(path):file_hash(path) for path in
        (PLAN,directory/'campaign-before.json',directory/'native-metrics-before.jsonl',paths[0],Path(before['producer_snapshot_manifest']))}
    old_manifest = _track(Path(before['producer_snapshot_manifest']))
    for spec in old_manifest.values():
        path = Path(spec['snapshot']); artifacts[str(path)] = file_hash(path)
    manifest = {}; pins = {}
    for original in [*before['code_pins'], str(ROOT/'src/episode_draft_recovery.py')]:
        original_path = Path(original); destination = directory/'producer-code'/original_path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(original_path.read_bytes()); destination.chmod(0o444)
        pins[original] = file_hash(destination); manifest[original] = {'snapshot':str(destination),'sha256':pins[original]}
    write_json(directory/'producer-code/manifest.json',manifest)
    rec = {'version':VERSION, 'campaign_before':str(directory/'campaign-before.json'),
        'native_metrics_before':str(directory/'native-metrics-before.jsonl'), 'artifacts':artifacts,
        'spent_local_seconds':before['local_seconds'], 'maximum_new_writer_requests':0,
        'maximum_new_asr_requests':0, 'derived_receipt_sha256':derived['receipt_sha256'],
        'parsed_sha256':fingerprint(derived['parsed']), 'created_utc':sr.now()}
    write_json(directory/'recovery.json',rec)
    conf = deepcopy(before); conf['input_hashes'].update(artifacts)
    conf['input_hashes'][str(directory/'recovery.json')] = file_hash(directory/'recovery.json')
    conf.update(status='prepared',native_attempt_recovery=str(directory/'recovery.json'),code_pins=pins,
        producer_snapshot_manifest=str(directory/'producer-code/manifest.json'),
        code_epochs=[_historical_epoch(before)])
    elapsed = time.monotonic()-start; conf['local_seconds'] += elapsed
    conf['rounds']['L1']['local_seconds'] += elapsed
    conf['rounds']['L1']['stages'].append({'name':'mechanical_native_attempt_registration','seconds':elapsed,'error':None,'finished_utc':sr.now()})
    validate_registration(folder,conf)
    write_json(folder/'campaign.json',conf)
    controller.check(folder)
    return {'status':conf['status'],'local_seconds':conf['local_seconds'],'new_writer_requests':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path)
    args=parser.parse_args()
    import fcntl, json
    with args.campaign.with_suffix('.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(register(args.campaign)))

if __name__ == '__main__':main()
