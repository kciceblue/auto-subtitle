"""Pure, source-bound overlap of saved VAD intervals with raw writer views.

Callers must independently validate the writer body and the saved preparation's
file/mono/runtime lineage. This module checks closed shapes, hash bindings and
exact interval arithmetic; it neither reads those files nor attests execution.
The detector record contains only ``mono: {frames, sample_rate}``, ``vad_speech``
and the four saved ``vad_options``. Its caller-supplied record hash uses
``workflow_state.fingerprint``. No observation is edited, copied into a fallback,
filtered, classified as incorrect, or assigned to a speaker.

All scope counts use the detector's sample grid. Integral counts are integers;
other values are reduced {numerator, denominator} objects. The detected fraction
uses only the in-domain span and is null when that span is empty. Out-of-domain
frames never become detector non-detection. No probabilities are reconstructed.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import re
from typing import Any

from src import compact_raw_evidence as compact
from src import evidence_context as ec
from src import raw_evidence_draft_requests as raw
from src.workflow_state import fingerprint

VERSION = 'voice-activity-evidence-1'
_VAD_OPTIONS = {'threshold': 0.5, 'min_speech_duration_ms': 0,
                'min_silence_duration_ms': 300, 'speech_pad_ms': 0}
_PROVENANCE_KEYS = {'short_preparation_sha256', 'original_preparation_sha256', 'mono_sha256',
                    'vad_model_sha256', 'vad_implementation_sha256', 'runtime_metadata_sha256',
                    'detector_record_sha256'}
_CAPABILITIES = {
    'detected_activity_is_speech_truth': False,
    'non_detection_establishes_speech_absence': False,
    'asr_correctness_verified': False,
    'speaker_identity_verified': False,
    'word_ownership_verified': False,
    'independent_recognizer_vote': False,
    'transformed_audio_detection_inferred': False,
    'observation_filtering_allowed': False,
    'native_detector_execution_independently_verified': False,
    'frame_probabilities_available': False,
}
_SIDEBAND_KEYS = {'version', 'bindings', 'provenance', 'detector', 'capabilities', 'views'}
_MEASURED_KINDS = {'original_transcript', 'raw', 'raw_early', 'raw_late', 'raw_short'}
_UNMEASURED_KINDS = {'bandit_dialogue', 'bandit_residual'}


def _same(left: Any, right: Any) -> bool:
    return fingerprint(left) == fingerprint(right)


def _exact(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {'numerator': value.numerator, 'denominator': value.denominator}


def _logical_body(body: dict, body_format: str) -> tuple[dict, dict[str, int]]:
    ec._require(body_format in ('raw', 'compact'), 'Unknown activity input body format')
    logical = compact.decode_body(body) if body_format == 'compact' else body
    coverage = raw.validate_writer_request({'instruction': raw.WRITER_INSTRUCTION,
                                          'body': logical, 'schema': raw.writer_schema()})
    return logical, coverage


def _check_record(record: dict, provenance: dict) -> None:
    ec._keys(record, {'mono', 'vad_speech', 'vad_options'})
    ec._keys(record['mono'], {'frames', 'sample_rate'})
    ec._require(all(type(value) is int and value > 0 for value in record['mono'].values()),
                'Invalid original-mono detector domain')
    ec._keys(record['vad_options'], set(_VAD_OPTIONS))
    ec._require(_same(record['vad_options'], _VAD_OPTIONS), 'Saved short VAD options changed')
    intervals = record['vad_speech']
    ec._require(isinstance(intervals, list), 'Saved VAD intervals must be a list')
    previous = 0
    for interval in intervals:
        ec._keys(interval, {'start', 'end'})
        ec._require(all(type(value) is int for value in interval.values()), 'VAD bounds must be integer frames')
        start, end = interval['start'], interval['end']
        ec._require(previous <= start < end <= record['mono']['frames'],
                    'VAD intervals overlap, are reordered, or exceed the detector domain')
        previous = end
    ec._keys(provenance, _PROVENANCE_KEYS)
    for value in provenance.values():
        ec._require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
                    'Invalid activity provenance hash')
    ec._require(provenance['detector_record_sha256'] == fingerprint(record), 'Saved detector record hash differs')


def _scope(interval: list[str], record: dict) -> dict:
    rate, domain_end = record['mono']['sample_rate'], record['mono']['frames']
    start, end = (Fraction(value) * rate for value in interval)
    total = end-start
    in_domain = max(Fraction(0), min(end, domain_end)-max(start, 0))
    detected = sum((max(Fraction(0), min(end, row['end'])-max(start, row['start']))
                    for row in record['vad_speech']), Fraction(0))
    ec._require(total > 0 and 0 <= detected <= in_domain <= total, 'Invalid activity overlap accounting')
    return {'interval_detector_frames': [_exact(start), _exact(end)],
            'total_detector_frames': _exact(total),
            'in_detector_domain_frames': _exact(in_domain),
            'out_of_detector_domain_frames': _exact(total-in_domain),
            'detected_frames': _exact(detected),
            'not_detected_in_domain_frames': _exact(in_domain-detected),
            'detected_fraction_of_in_domain': _exact(detected/in_domain) if in_domain else None}


def _build(body: dict, record: dict, provenance: dict, body_format: str) -> tuple[dict, dict[str, int]]:
    logical, coverage = _logical_body(body, body_format)
    _check_record(record, provenance)
    views = []; applicable = cores = 0
    for view in logical['source_evidence']['views']:
        kind = view['kind']
        item = {'view_id': view['id'], 'owner_id': view['owner_id'], 'kind': kind,
                'view_sample_rate': view['sample_rate']}
        if kind in _MEASURED_KINDS:
            crop = _scope(view['crop_seconds'], record)
            core = _scope(view['core_seconds'], record) if view['core_seconds'] is not None else None
            item.update(status='original_mono_detector_applicable', crop=crop, core=core)
            applicable += 1; cores += core is not None
        else:
            ec._require(kind in _UNMEASURED_KINDS, 'Unknown activity view kind')
            item.update(status='not_measured_on_this_transform', crop=None, core=None)
        views.append(item)
    result = {
        'version': VERSION,
        'bindings': {'body_format': body_format, 'input_body_sha256': fingerprint(body),
                     'logical_raw_body_sha256': fingerprint(logical)},
        'provenance': deepcopy(provenance),
        'detector': {'scope': 'original_mono', 'mono': deepcopy(record['mono']),
                     'vad_options': deepcopy(record['vad_options']),
                     'interval_count': len(record['vad_speech']),
                     'detected_frames': sum(row['end']-row['start'] for row in record['vad_speech']),
                     'count_unit': 'detector_sample_frames',
                     'fraction_denominator': 'in_detector_domain_frames'},
        'capabilities': deepcopy(_CAPABILITIES), 'views': views,
    }
    return result, {**coverage, 'original_mono_views': applicable,
                    'unmeasured_transform_views': len(views)-applicable, 'core_intervals': cores}


def build_sideband(body: dict, detector_record: dict, provenance: dict, *, body_format: str = 'raw') -> dict:
    """Return a separate numeric sideband without mutating body or saved records.

    ``body`` is an independently validated A raw or B compact body. File hashes
    in ``provenance`` are caller attestations, not files opened by this builder.
    No intervals, non-detections, observations or transformed views are dropped.
    """
    try:
        return _build(body, detector_record, provenance, body_format)[0]
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed voice-activity inputs') from error


def validate_sideband(sideband: dict, body: dict, detector_record: dict, provenance: dict,
                      *, body_format: str = 'raw') -> dict[str, int]:
    """Recompute every field from independent inputs; reject valid-looking edits.

    This verifies the association with the supplied inputs. Authentication of
    those inputs against source/native artifacts remains the caller's job.
    """
    ec._keys(sideband, _SIDEBAND_KEYS)
    try:
        expected, coverage = _build(body, detector_record, provenance, body_format)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed voice-activity inputs') from error
    ec._require(_same(sideband, expected), 'Activity sideband differs from independently supplied inputs')
    return coverage
