"""Confirmed contextual usability milestone; never audio-verified release."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from src.contextual_review import validate_receipt
from src.contextual_review_contract import ASSESSMENT_SCOPE, BENCHMARK, require
from src.release import (COHERENCE_TIMING_WARNINGS, _valid_writer_model,
                         deliverables, inspect_deliverables)
from src.config import MEDIA_EXTENSIONS


def prepare_assessment(unit: Path, path: Path, *, writer_model: str) -> dict:
    require(_valid_writer_model(writer_model), 'Identify the actual local writer')
    require(not path.resolve().is_relative_to((unit/'final').resolve()),
            'Keep assessment outside final/')
    files = deliverables(unit)
    record = dict(benchmark=BENCHMARK, assessment_scope=ASSESSMENT_SCOPE,
                  unit=unit.name, files=files, writer_model=writer_model,
                  writer_execution='local', score=None, contextual_reviews=[],
                  source_accuracy_verified=False, playback_verified=False,
                  automatic_findings=inspect_deliverables(unit, files))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return record


def _pairs(files: dict[str, str]) -> dict[str, str]:
    """Map each translated deliverable to the source owned by its media."""
    pairs = {}
    media = [Path(name) for name in files if Path(name).suffix.lower() in MEDIA_EXTENSIONS]
    sources = {path.with_suffix('.srt').as_posix() for path in media}
    for path in media:
        prefix = path.stem + '.'
        for name in files:
            target = Path(name)
            if (name not in sources and target.suffix.lower() == '.srt'
                    and target.parent == path.parent and target.stem.startswith(prefix)
                    and re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*',
                                     target.stem[len(prefix):])):
                pairs[name] = path.with_suffix('.srt').as_posix()
    return pairs


def validate_assessment(unit: Path, assessment: dict) -> list[str]:
    """Validate current artifacts and two actual score-only reviews per target.

    Confidence is reported, not a calibrated accuracy estimate or pass criterion.
    Timing warnings remain visible; this scope cannot approve release packaging.
    """
    try:
        require(isinstance(assessment, dict), 'Assessment must be an object')
        require(assessment.get('benchmark') == BENCHMARK
                and assessment.get('assessment_scope') == ASSESSMENT_SCOPE,
                'Expected contextual subtitle benchmark v4')
        require(assessment.get('unit') == unit.name, 'Assessment unit changed')
        require(_valid_writer_model(assessment.get('writer_model'))
                and assessment.get('writer_execution') == 'local',
                'A local writer independent of the evaluator is required')
        require(assessment.get('source_accuracy_verified') is False
                and assessment.get('playback_verified') is False,
                'Contextual review cannot certify audio truth or playback')
        files = deliverables(unit)
        require(assessment.get('files') == files, 'Final files changed after assessment')
        findings = inspect_deliverables(unit, files)
        blocking = [f for f in findings if f['kind'] not in COHERENCE_TIMING_WARNINGS]
        require(not blocking, 'Structural checks failed: ' + ', '.join(f['id'] for f in blocking))
        pairs = _pairs(files)
        require(bool(pairs), 'No paired translated subtitles')
        reviews = assessment.get('contextual_reviews')
        require(isinstance(reviews, list) and len(reviews) == len(pairs),
                'Every translated file needs primary and confirmation reviews')
        seen = set(); dispatches = set(); scores = []
        for evidence in reviews:
            require(isinstance(evidence, dict)
                    and set(evidence) == {'target', 'manifest', 'primary', 'confirmation',
                                         'primary_receipt_hashes', 'confirmation_receipt_hashes'},
                    'Invalid contextual evidence mapping')
            target = evidence['target']
            require(isinstance(target, str) and target in pairs and target not in seen,
                    'Unknown or duplicate reviewed target')
            seen.add(target)
            primary = validate_receipt(evidence['primary'], evidence['manifest'])
            confirmation = validate_receipt(evidence['confirmation'], evidence['manifest'])
            for role, receipt in (('primary', primary), ('confirmation', confirmation)):
                expected = evidence[role + '_receipt_hashes']
                require(isinstance(expected, dict) and bool(expected)
                        and all(isinstance(name, str) and isinstance(digest, str)
                                and re.fullmatch(r'[0-9a-f]{64}', digest) is not None
                                for name, digest in expected.items())
                        and expected == receipt.get('receipt_hashes'),
                        'Contextual receipt set changed after assessment')
            require(primary['inputs'] == confirmation['inputs'], 'Confirmation reviewed different inputs')
            require(datetime.fromisoformat(confirmation['started_utc'])
                    >= datetime.fromisoformat(primary['finished_utc']),
                    'Confirmation must be a fresh dispatch after primary completion')
            for receipt in (primary, confirmation):
                require(receipt['dispatch_id'] not in dispatches, 'Reviewer dispatch reused')
                dispatches.add(receipt['dispatch_id'])
                require(receipt['score'] >= 4 and receipt['contextual_usability_pass'] is True,
                        'Primary and confirmation must both score at least four')
                require(receipt['inputs']['target']['sha256'] == files[target]
                        and receipt['inputs']['source']['sha256'] == files[pairs[target]],
                        'Reviewed subtitles differ from final source or translation')
                scores.append(receipt['score'])
        require(type(assessment.get('score')) is int and assessment['score'] == min(scores),
                'Milestone score must be the conservative minimum of all confirmed reviews')
        require(deliverables(unit) == files, 'Final files changed during contextual gate validation')
        return []
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        return [str(error)]
