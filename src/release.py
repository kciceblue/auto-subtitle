"""Independent release assessment bound to the exact deliverable contents.

Pipeline completion and handoff for review do not constitute release approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.config import MEDIA_EXTENSIONS
from src.quality import seconds, validate_blocks
from src.translate import SrtBlock
from src.workflow_state import file_hash, read_json, write_json
from src.scalar_review_contract import (PROMPT as SCALAR_PROMPT, SCHEMA as SCALAR_SCHEMA,
                                        validate_answer as validate_scalar_answer)

BENCHMARK_VERSION = 'subtitle-quality-v3'
WRITER_MODEL = 'qwen3.8-27b-dflash'
ALLOWED_REVIEWER_MODELS = frozenset({'gpt-6-astra', 'fable-5.1'})
REVIEWER_SCOPE = 'evaluation_only'
REVIEWER_FEEDBACK = 'score_only'
SOURCE_VERIFIED = 'source_verified'
TARGET_COHERENCE = 'target_coherence'
CONFIRMED_SCALAR = 'confirmed_scalar_v1'
CHECKS = ('source_checked_in_full', 'meaning_and_facts', 'completeness',
          'names_and_roles', 'timing_and_readability', 'language')
COHERENCE_CHECKS = ('target_checked_in_full', 'logical_coherence',
                    'natural_language', 'context_consistency', 'text_readability')
COHERENCE_TIMING_WARNINGS = frozenset({'short-cue', 'long-cue'})


def _valid_writer_model(model: object) -> bool:
    return (isinstance(model, str) and bool(model.strip())
            and model.strip().casefold() not in ALLOWED_REVIEWER_MODELS)


def deliverables(unit: Path) -> dict[str, str]:
    final = unit / 'final'
    if final.is_symlink():
        raise ValueError(f'Release deliverable cannot be a symlink: {final}')
    if not final.is_dir():
        raise ValueError(f'{unit} needs an organized final/ directory')
    files = {}
    for path in sorted(final.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Release deliverable cannot be a symlink: {path}')
        if path.is_file():
            files[path.relative_to(unit).as_posix()] = file_hash(path)
    if not files or not any(Path(name).suffix.lower() == '.srt' for name in files):
        raise ValueError('Release requires subtitle deliverables')
    return files


def _read_release_srt(path: Path) -> list[SrtBlock]:
    """Read every chunk strictly; release must not silently discard broken cues."""
    raw = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    blocks = []
    timestamp = r'[0-9]{2}:[0-5][0-9]:[0-5][0-9],[0-9]{3}'
    for number, chunk in enumerate(re.split(r'\n\s*\n', raw.strip()), 1):
        lines = chunk.strip().splitlines()
        if (len(lines) < 3 or not re.fullmatch(r'[0-9]+', lines[0].strip())
                or not re.fullmatch(timestamp + ' --> ' + timestamp, lines[1].strip())):
            raise ValueError(f'Malformed SRT chunk {number}')
        blocks.append(SrtBlock(int(lines[0].strip()), lines[1].strip(),
                               '\n'.join(line.strip() for line in lines[2:])))
    validate_blocks(blocks)
    return blocks


def inspect_deliverables(unit: Path, files: dict[str, str]) -> list[dict]:
    findings = []

    def add(name, line, kind, detail):
        findings.append(dict(id=f'{name}:{line}:{kind}', file=name, line=line,
                             kind=kind, detail=detail))

    subtitles = {name for name in files if Path(name).suffix.lower() == '.srt'}
    parsed = {}
    for name in sorted(subtitles):
        try:
            blocks = _read_release_srt(unit / name)
        except (ValueError, OSError) as exc:
            add(name, 0, 'invalid-srt', str(exc))
            continue
        parsed[name] = blocks
        previous_end = 0.0
        for block in blocks:
            start, end = map(seconds, block.ts_line.split('-->'))
            if end <= start:
                add(name, block.index, 'nonpositive', 'Cue end must follow its start')
            if start < previous_end:
                add(name, block.index, 'overlap', 'Cue overlaps the preceding cue')
            if end-start < .4:
                add(name, block.index, 'short-cue', 'Display duration below 0.4 seconds')
            if end-start > 10:
                add(name, block.index, 'long-cue', 'Display duration above 10 seconds')
            previous_end = end

    media_names = {name for name in files if Path(name).suffix.lower() in MEDIA_EXTENSIONS}
    source_names = {Path(name).with_suffix('.srt').as_posix() for name in media_names}
    if not media_names:
        add('final', 0, 'missing-media', 'Source media is required for a release package')
    assigned = subtitles & source_names
    for name in sorted(media_names):
        media = Path(name)
        source_name = media.with_suffix('.srt').as_posix()
        prefix = media.stem + '.'
        translations = []
        for candidate in sorted(subtitles - source_names):
            target = Path(candidate)
            if target.parent != media.parent or not target.stem.startswith(prefix):
                continue
            language = target.stem[len(prefix):]
            # Pipeline deliverables use a language code (zh, en, zh-Hant, ...).
            # Snapshots, utterance sidecars and another media's source are not
            # translated deliverables even when their filenames share a prefix.
            if (not language.lower().startswith('pre-')
                    and re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*', language)):
                translations.append(candidate)
        assigned.update(translations)
        if source_name not in subtitles:
            if translations:
                add(name, 0, 'missing-source', 'Translated subtitles have no paired source subtitles')
            else:
                add(name, 0, 'no-speech-subtitles', 'No source subtitles; reviewer must verify an empty/rest track')
        elif not translations:
            add(name, 0, 'missing-translation', 'Source subtitles have no translated counterpart')
        elif source_name in parsed:
            source = parsed[source_name]
            for target_name in translations:
                if target_name not in parsed:
                    continue  # The strict parser already reported the invalid file.
                target = parsed[target_name]
                if len(source) != len(target) or any(a.index != b.index or a.ts_line != b.ts_line for a, b in zip(source, target)):
                    add(target_name, 0, 'source-target-alignment', 'Source and translation IDs/timestamps differ')
    for name in sorted(subtitles - assigned):
        add(name, 0, 'orphan-subtitles', 'Subtitle is not a source or a language-code translation paired with media')
    return findings


def prepare_assessment(unit: Path, path: Path, *, writer_model: str = WRITER_MODEL,
                       assessment_scope: str = SOURCE_VERIFIED) -> dict:
    if not _valid_writer_model(writer_model):
        raise ValueError('Identify a local writer model; stronger evaluator models cannot write subtitles')
    if assessment_scope not in (SOURCE_VERIFIED, TARGET_COHERENCE):
        raise ValueError('Assessment scope must be source_verified or target_coherence')
    if path.resolve().is_relative_to((unit / 'final').resolve()):
        raise ValueError('Keep the assessment outside final/ so its own hash cannot invalidate the file manifest')
    if path.exists():
        raise FileExistsError(f'Preserve the existing assessment: {path}')
    files = deliverables(unit)
    record = dict(benchmark=BENCHMARK_VERSION, unit=unit.name, files=files, score=None,
                  assessment_scope=assessment_scope,
                  writer_model=writer_model, writer_execution='local',
                  reviewer=dict(name='', kind='independent_model', model='',
                                scope=REVIEWER_SCOPE, feedback_to_writer=REVIEWER_FEEDBACK,
                                independent_of_writer=False),
                  reviewed_at='', checks={name: False for name in CHECKS},
                  coherence_checks={name: False for name in COHERENCE_CHECKS},
                  coherence_open_issues=[], coherence_polish_notes=[],
                  open_issues=[], automatic_findings=inspect_deliverables(unit, files),
                  finding_exceptions={})
    write_json(path, record)
    return record


def validate_assessment(unit: Path, assessment: dict) -> list[str]:
    """Validate the strict score-six release gate, never the coherence milestone."""
    return _validate_assessment(unit, assessment, assessment_scope=SOURCE_VERIFIED)


def validate_coherence_assessment(unit: Path, assessment: dict) -> list[str]:
    """Validate a score-four/five Chinese-only review without approving release."""
    return _validate_assessment(unit, assessment, assessment_scope=TARGET_COHERENCE)


def coherence_warnings(unit: Path) -> list[dict]:
    """Current duration findings need playback review but do not block score four."""
    return [finding for finding in inspect_deliverables(unit, deliverables(unit))
            if finding['kind'] in COHERENCE_TIMING_WARNINGS]


def _scalar_file(unit: Path, value: object) -> Path:
    """Resolve an evidence path without accepting symlinks in any component."""
    if not isinstance(value, str) or not value or '..' in Path(value).parts:
        raise ValueError('Invalid scalar evidence path')
    path = Path(value)
    if not path.is_absolute():
        path = unit / path
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Scalar evidence and candidate paths cannot contain symlinks')
    if not path.is_file():
        raise ValueError('Missing scalar evidence or candidate file')
    return path


def _scalar_json(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate scalar JSON field')
            result[key] = value
        return result
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('Scalar receipt must be a JSON object')
    return value


def _scalar_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Missing scalar dispatch date')
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Scalar dispatch dates require a timezone')
    return result


def _validate_confirmed_scalars(unit: Path, assessment: dict, files: dict[str, str]) -> dict[Path, str]:
    """Validate existing four-field reviews directly; create no check attestations.

    Receipt hashes and dispatch headers are local provenance, not a provider signature.
    Return all evidence hashes for a second check after structural validation.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    tracked = {}

    def read_bound(path, digest=None):
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(digest is None or actual == digest, 'Scalar evidence hash mismatch')
        require(path not in tracked or tracked[path] == actual, 'Scalar evidence changed during validation')
        tracked[path] = actual
        return raw

    require(type(assessment.get('score')) is int and assessment['score'] == 4,
            'Confirmed scalar evidence requires score4 exactly')
    # Source subtitles must not be mistaken for targets, including media names
    # whose stems themselves contain a language code.
    sources = {Path(name).with_suffix('.srt').as_posix() for name in files
               if Path(name).suffix.lower() in MEDIA_EXTENSIONS}
    targets = {name for name in files if Path(name).suffix.lower() == '.srt'} - sources
    reviews = assessment.get('scalar_reviews')
    require(bool(targets) and isinstance(reviews, dict) and set(reviews) == targets,
            'Scalar evidence must cover every final translated subtitle file exactly')
    require(all(re.search(r'\.zh(?:-[A-Za-z0-9]{1,8})*\.srt$', name, re.IGNORECASE)
                for name in targets), 'Confirmed scalar evidence supports Chinese subtitle targets only')
    checks = assessment.get('coherence_checks', {})
    require(isinstance(checks, dict) and all(value is False for value in checks.values()),
            'Scalar evidence cannot assert separate coherence checks')
    used_dispatches = set()
    confirmations = []
    for relative in sorted(targets):
        target = _scalar_file(unit, relative)
        raw = read_bound(target, files[relative])
        blocks = _read_release_srt(target)
        require(not any(re.fullmatch(r'\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->.*', line)
                        for block in blocks for line in block.text.splitlines()),
                'Embedded cue timestamp in scalar candidate')
        count = len(blocks)
        pair = reviews[relative]
        require(isinstance(pair, dict) and set(pair) == {'primary', 'confirmation'},
                'Require exactly primary and confirmation scalar receipt sets')
        previous_end = None
        original_candidate = None
        for role in ('primary', 'confirmation'):
            receipt_set = pair[role]
            require(isinstance(receipt_set, dict) and set(receipt_set) == {'assessment', 'response', 'run'},
                    'Invalid scalar receipt set')
            paths, payloads = {}, {}
            names = {'assessment': 'coherence-assessment.json', 'response': 'review-response.json',
                     'run': 'review-run.json'}
            for kind in names:
                spec = receipt_set[kind]
                require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}
                        and isinstance(spec['sha256'], str)
                        and re.fullmatch(r'[0-9a-f]{64}', spec['sha256']) is not None,
                        'Scalar receipts require an exact path and SHA256')
                path = _scalar_file(unit, spec['path'])
                require(path.name == names[kind] and not path.is_relative_to((unit / 'final').absolute()),
                        'Keep original scalar receipt files outside final')
                paths[kind] = path
                payloads[kind] = _scalar_json(read_bound(path, spec['sha256']))
            require(len({path.parent for path in paths.values()}) == 1,
                    'Scalar receipt set must belong to one dispatch directory')
            record, answer, run = (payloads[key] for key in ('assessment', 'response', 'run'))
            validate_scalar_answer(answer)
            require(answer['score'] == 4 and answer['target_only_coherence_pass'] is True,
                    'Both scalar reviews must pass at score4')
            require(all(key in record and type(record[key]) is type(answer[key]) and record[key] == answer[key]
                        for key in SCALAR_SCHEMA['required']), 'Scalar receipt disagrees with its raw response')
            require(record.get('rubric_version') == BENCHMARK_VERSION
                    and record.get('assessment_kind') == 'scalar_review'
                    and record.get('release_gate_checked') is False,
                    'Invalid scalar benchmark or assessment kind')
            identity = record.get('reviewer')
            require(isinstance(identity, dict) and isinstance(identity.get('model'), str)
                    and identity['model'] in ALLOWED_REVIEWER_MODELS
                    and identity.get('scope') == REVIEWER_SCOPE
                    and identity.get('feedback_to_writer') == REVIEWER_FEEDBACK
                    and identity.get('independent_of_writer') is True
                    and identity.get('prior_exposure') is False
                    and identity.get('dispatch') == 'explicit ephemeral codex exec --model',
                    'Invalid independent scalar reviewer provenance')
            require(record.get('sha256') == files[relative]
                    and all(type(record.get(key)) is int and record[key] == count
                            for key in ('cue_count', 'cues_reviewed'))
                    and type(record.get('coverage')) in (int, float) and record['coverage'] == 1.0,
                    'Scalar receipt does not cover the exact current candidate')
            origin = _scalar_file(unit, record.get('file'))
            require(read_bound(origin, files[relative]) == raw, 'Reviewed candidate bytes differ from final')
            require(original_candidate is None or origin == original_candidate,
                    'Confirmation must review the same original candidate path')
            original_candidate = origin
            require(_scalar_file(unit, run.get('input_file')) == origin
                    and run.get('input_sha256') == files[relative], 'Scalar dispatch input binding mismatch')
            elapsed = run.get('seconds')
            require(type(run.get('exit_code')) is int and run['exit_code'] == 0
                    and type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
                    and run.get('scope') == REVIEWER_SCOPE
                    and run.get('feedback_to_writer') == REVIEWER_FEEDBACK
                    and run.get('explicit_generated_text_export_authorized') is True,
                    'Scalar dispatch did not complete successfully in score-only scope')
            started = _scalar_time(run.get('started_utc'))
            require(previous_end is None or started >= previous_end,
                    'Confirmation must start after the primary dispatch completed')
            previous_end = started + timedelta(seconds=elapsed)
            stat = paths['run'].stat()
            dispatch_id = (stat.st_dev, stat.st_ino)
            require(dispatch_id not in used_dispatches, 'Cannot reuse a scalar dispatch receipt')
            used_dispatches.add(dispatch_id)
            command = run.get('command')
            require(isinstance(command, list) and all(isinstance(arg, str) for arg in command)
                    and command[:2] == ['codex', 'exec'] and command[-1:] == ['-']
                    and command.count('--ephemeral') == 1, 'Require a fresh explicit scalar evaluator dispatch')

            def option(flag):
                require(command.count(flag) == 1 and command.index(flag) + 1 < len(command),
                        'Invalid scalar evaluator command option')
                return command[command.index(flag) + 1]

            require(option('--model') == identity['model'] and option('--sandbox') == 'read-only'
                    and option('--cd') == '/tmp', 'Scalar evaluator command identity or isolation mismatch')
            require(_scalar_file(unit, option('--output-last-message')) == paths['response'],
                    'Scalar response is not the recorded dispatch output')
            headers = run.get('observed_identity_headers')
            require(isinstance(headers, dict) and headers.get('model') == identity['model']
                    and isinstance(headers.get('provider'), str) and bool(headers['provider'].strip())
                    and (identity['model'] != 'gpt-6-astra' or headers['provider'] == 'openai'),
                    'Scalar dispatch observed identity mismatch')
            prompt_path = _scalar_file(unit, str(paths['run'].parent / 'review-prompt.txt'))
            prompt = read_bound(prompt_path, run.get('prompt_sha256'))
            require(isinstance(run.get('prompt_sha256'), str)
                    and prompt == SCALAR_PROMPT.encode('utf-8') + raw,
                    'Scalar prompt must be the unchanged Chinese-only score contract and current candidate')
            schema_path = _scalar_file(unit, option('--output-schema'))
            require(schema_path == paths['run'].parent / 'review-schema.json'
                    and _scalar_json(read_bound(schema_path)) == SCALAR_SCHEMA,
                    'Scalar output schema must retain exactly four score-only fields')
            if role == 'confirmation':
                confirmations.append((started, identity['model']))
    latest = max(confirmations)
    require(_scalar_time(assessment.get('reviewed_at')) == latest[0]
            and assessment.get('reviewer', {}).get('model') == latest[1],
            'Assessment reviewer and date must identify the latest confirmation dispatch start')
    return tracked


def _validate_assessment(unit: Path, assessment: dict, *, assessment_scope: str) -> list[str]:
    errors = []
    if not isinstance(assessment, dict):
        return ['Missing or invalid independent assessment']
    if assessment.get('benchmark') != BENCHMARK_VERSION:
        errors.append('Missing or outdated benchmark version')
    if assessment.get('assessment_scope') != assessment_scope:
        errors.append(f'Assessment scope must be {assessment_scope}; another gate cannot substitute')
    if assessment.get('unit') != unit.name:
        errors.append('Assessment belongs to another work unit')
    try:
        files = deliverables(unit)
    except (ValueError, OSError) as exc:
        return [*errors, str(exc)]
    if assessment.get('files') != files:
        errors.append('Deliverable content changed or file set differs; revalidate the current files')
    score = assessment.get('score')
    coherent = assessment_scope == TARGET_COHERENCE
    evidence_kind = assessment.get('coherence_evidence_kind')
    scalar = evidence_kind == CONFIRMED_SCALAR
    scalar_files = {}
    if evidence_kind not in (None, CONFIRMED_SCALAR):
        errors.append('Unknown coherence evidence kind')
    if scalar and not coherent:
        errors.append('Confirmed scalar evidence cannot approve a source-verified assessment')
    if not scalar and 'scalar_reviews' in assessment:
        errors.append('Scalar receipts require explicit confirmed_scalar_v1 evidence kind')
    if coherent:
        if type(score) not in (int, float) or not 4 <= score < 6:
            errors.append('A target-coherence assessment must assign a score from 4 up to, but below, 6')
    elif type(score) not in (int, float) or not 6 <= score <= 10:
        errors.append('A release assessment must assign a score from 6 to 10')
    reviewer = assessment.get('reviewer', {})
    if not isinstance(reviewer, dict):
        reviewer = {}
    if not isinstance(reviewer.get('name'), str) or not reviewer['name'].strip():
        errors.append('Identify the independent reviewer')
    writer_model = assessment.get('writer_model')
    if not _valid_writer_model(writer_model):
        errors.append('This benchmark requires an identified local writer model, separate from stronger evaluators')
    if assessment.get('writer_execution') != 'local':
        errors.append('The writer must be explicitly recorded as writer_execution=local')
    if reviewer.get('kind') != 'independent_model' or reviewer.get('independent_of_writer') is not True:
        errors.append('Assessment requires a separate independent model reviewer; writer QA or human review alone is insufficient')
    reviewer_model = reviewer.get('model')
    if not isinstance(reviewer_model, str) or reviewer_model not in ALLOWED_REVIEWER_MODELS:
        errors.append('Release reviewer model must be gpt-6-astra or fable-5.1; the local writer cannot approve its output')
    if reviewer.get('scope') != REVIEWER_SCOPE:
        errors.append('The stronger reviewer must be evaluation_only and must not write translations or repairs')
    if reviewer.get('feedback_to_writer') != REVIEWER_FEEDBACK:
        errors.append('Reviewer feedback to the writer must be score_only; audit findings cannot guide subtitle repairs')
    try:
        reviewed = datetime.fromisoformat(assessment.get('reviewed_at', ''))
        if reviewed.tzinfo is None:
            raise ValueError('Timezone required')
    except (TypeError, ValueError):
        errors.append('Supply the actual review date with a timezone')
    if coherent and scalar:
        try:
            scalar_files = _validate_confirmed_scalars(unit, assessment, files)
        except (ValueError, OSError, TypeError, OverflowError, KeyError, AttributeError):
            # Do not echo potentially sensitive evidence values or model text.
            errors.append('Invalid confirmed scalar evidence; verify receipts, coverage, identity, dates and bindings')
    else:
        checks = assessment.get('coherence_checks' if coherent else 'checks', {})
        required = COHERENCE_CHECKS if coherent else CHECKS
        if not isinstance(checks, dict) or any(checks.get(name) is not True for name in required):
            errors.append('Every Chinese-only coherence check must pass' if coherent
                          else 'Every full-source validation check must pass')
    # An absent scalar-only issue inventory makes no independent attestation.
    # Any explicitly recorded blocking coherence issue still blocks both paths.
    if assessment.get('coherence_open_issues', [] if coherent and scalar else None) != []:
        errors.append('All blocking Chinese-only coherence issues must be resolved')
    # Optional in v3 for backward compatibility. These notes attest to minor
    # weaknesses that are not obvious and do not impair understanding. They do
    # not waive a failed check or a blocking issue, and never enter writer input.
    polish = assessment.get('coherence_polish_notes', [])
    if (not isinstance(polish, list)
            or any(not isinstance(note, dict) or set(note) != {'severity', 'note'}
                   or note.get('severity') != 'non_obvious_polish'
                   or not isinstance(note.get('note'), str) or not note['note'].strip()
                   for note in polish)):
        errors.append('Coherence polish notes must explicitly identify non-obvious minor polish')
    elif not coherent and polish:
        errors.append('All recorded target-language issues must be resolved before source-verified release')
    if not coherent and assessment.get('open_issues') != []:
        errors.append('All errors and unresolved questions must be resolved before release')
    exceptions = assessment.get('finding_exceptions', {})
    if not isinstance(exceptions, dict):
        exceptions = {}
    for finding in inspect_deliverables(unit, files):
        if coherent and finding['kind'] in COHERENCE_TIMING_WARNINGS:
            continue  # Exposed by coherence_warnings; full playback is a six-level check.
        reason = exceptions.get(finding['id'])
        # A reviewer may explain an intentional brief interjection or similar
        # false positive. Broken SRT structure cannot be waived.
        if (finding['kind'] in {'invalid-srt', 'nonpositive', 'overlap', 'missing-media', 'missing-source', 'missing-translation', 'source-target-alignment', 'orphan-subtitles'}
                or not isinstance(reason, str) or not reason.strip()):
            errors.append(f"Unresolved {finding['id']}: {finding['detail']}")
    if coherent and scalar:
        try:
            if deliverables(unit) != files or any(
                    file_hash(_scalar_file(unit, str(path))) != digest
                    for path, digest in scalar_files.items()):
                errors.append('Confirmed scalar evidence or deliverables changed during validation')
        except (ValueError, OSError):
            errors.append('Confirmed scalar evidence or deliverables changed during validation')
    return errors


def release_unit(unit: Path, assessment_path: Path, destination: Path = Path('released')) -> Path:
    """Copy the validated final files and assessment to a local release folder + zip."""
    assessment = read_json(assessment_path)
    errors = validate_assessment(unit, assessment)
    if errors:
        raise ValueError('Release blocked: ' + '; '.join(errors[:12]))
    if destination.resolve().is_relative_to((unit / 'final').resolve()):
        raise ValueError('Release destination cannot be inside the source final/ directory')
    destination.mkdir(parents=True, exist_ok=True)
    target, archive = destination / unit.name, destination / f'{unit.name}.zip'
    if target.exists() or archive.exists():
        raise FileExistsError(f'Release already exists: {target} / {archive}')
    with tempfile.TemporaryDirectory(prefix='.release-', dir=destination) as directory:
        stage = Path(directory) / unit.name
        shutil.copytree(unit / 'final', stage / 'final')
        # Validate the staged bytes too, closing changes during the copy.
        errors = validate_assessment(stage, assessment)
        if errors:
            raise ValueError('Release changed while packaging: ' + '; '.join(errors[:12]))
        write_json(stage / 'release-assessment.json', assessment)
        staged_zip = Path(shutil.make_archive(str(Path(directory) / unit.name), 'zip',
                                             root_dir=directory, base_dir=unit.name))
        stage.replace(target)
        try:
            staged_zip.replace(archive)
        except OSError:
            shutil.rmtree(target)
            raise
    return target
