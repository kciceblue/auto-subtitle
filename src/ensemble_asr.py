"""Concurrent independent ASR, bounded acoustic recovery and timed cue rebuilding.

This entire worker exits before the 27B translator is loaded. Model families run
on independent raw-audio windows; no recognizer inherits Whisper's SRT cuts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from src.asr_consensus import (aligned_tokens, audio_windows, choose_transcript,
                               compact, phrase_grade, project_phrase, similarity,
                               subtitle_groups, suspicious_text, restore_sentence_boundaries, utterance_groups)
from src.workflow_state import StageState, artifact_path, file_hash, fingerprint, write_json

logger = logging.getLogger(__name__)
ALIGNER = Path.home() / 'HF/asr-models/Qwen3-ForcedAligner-0.6B'


class Recognizers:
    def __init__(self, config: dict, batch_size: int):
        import torch
        from src.warden import ensure_gpu_headroom
        from src.asr import load_model
        from src.adjudicate import Qwen3Q, ZipformerN
        from src.config import TranscribeConfig
        torch.set_num_threads(4)
        ensure_gpu_headroom(required_gb=16, hard_floor_gb=12,
                            admin_url=config['warden_admin_url'],
                            enabled=config['unload_warden_before_asr'], caller='concurrent ASR')
        cfg = TranscribeConfig(**config)
        cfg.unload_warden_before_asr = False
        self.w = load_model(cfg)
        self.q = Qwen3Q(unload_warden=False, compute_type="bfloat16")
        self.q._asr.max_new_tokens = 1024
        self.n = ZipformerN(num_threads=4, decoder_precision="fp32")
        self.batch_size = batch_size
        self.config = cfg
        self.aligner = None

    def recognize(self, audio, sr: int, windows, hotwords: list[str]):
        import numpy as np
        clips = [audio[int(w.start * sr):int(w.end * sr)] for w in windows]
        timings = {}
        epoch = time.monotonic()

        def worker(kind):
            started = time.monotonic()
            rows = [''] * len(windows)
            if kind == 'w':
                from faster_whisper import BatchedInferencePipeline
                model = BatchedInferencePipeline(self.w)
                # Explicit contiguous clips avoid global VAD concatenation/restoration.
                # The batch API asserts total clip duration <= audio duration.
                # Pack overlapping source windows into disjoint private slots;
                # ownership is mapped back below, and final alignment uses raw audio.
                offsets = np.cumsum([0] + [len(clip) for clip in clips])
                packed = np.concatenate([*clips, np.zeros(sr // 10, dtype=np.float32)])
                segments, _ = model.transcribe(
                    packed, language=self.config.language or 'ja',
                    beam_size=self.config.beam_size, batch_size=self.batch_size,
                    clip_timestamps=[{'start':int(a)/sr, 'end':int(b)/sr}
                                     for a,b in zip(offsets,offsets[1:])],
                    vad_filter=False, word_timestamps=True,
                    hotwords=' '.join(hotwords) or None,
                )
                for segment in segments:
                    owner = min(range(len(windows)),
                                key=lambda i: abs(int(offsets[i] / sr * self.w.frames_per_second) - segment.seek))
                    rows[owner] += segment.text.strip()
            elif kind == 'q':
                for start in range(0, len(clips), self.batch_size):
                    batch = clips[start:start + self.batch_size]
                    context = "、".join(hotwords) if self.config.qwen_asr_hotwords else ""
                    values = self.q.transcribe_batch(sr, batch, context=context)
                    for offset, value in enumerate(values):
                        rows[start + offset] = value if np.any(batch[offset]) else ''
                    logger.info('Qwen-ASR windows %d/%d', min(start+self.batch_size,len(clips)),len(clips))
            else:
                for i, clip in enumerate(clips):
                    rows[i] = self.n.transcribe(sr, clip) if np.any(clip) else ''
            finished = time.monotonic()
            timings[kind] = dict(start=started-epoch, end=finished-epoch, seconds=finished-started)
            logger.info('%s independent recognition complete: %d windows in %.2fs', kind,len(rows),finished-started)
            return kind, rows

        outputs = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(worker,k) for k in ('w','q','n')]):
                kind, rows = future.result()
                outputs[kind] = rows
        return [{k:outputs[k][i] for k in outputs} for i in range(len(windows))], timings

    def align(self, clips, texts, sr):
        if self.aligner is None:
            import torch
            from qwen_asr import Qwen3ForcedAligner
            self.aligner = Qwen3ForcedAligner.from_pretrained(str(ALIGNER),
                                                           dtype=torch.bfloat16, device_map='cuda')
        return self.aligner.align(audio=[(clip,sr) for clip in clips], text=texts, language='Japanese')



def select_window_source(candidates: dict[str, str], policy: str) -> dict:
    """Choose raw text without treating the two Reazon models as two votes."""
    if policy not in {"consensus", "qwen", "nemo"}:
        raise ValueError(f"Unknown ensemble source policy: {policy}")
    original = {k: v for k, v in candidates.items() if k in {'w', 'q', 'n'}}
    selected, score = choose_transcript(original)
    preferred_q, preferred_r, reason = False, False, None
    if policy in {'qwen', 'nemo'}:
        q = original.get('q', '')
        if not compact(q):
            reason = 'Original-audio Qwen output is empty or punctuation-only; using original consensus'
        elif suspicious_text(q):
            reason = 'Visible suspicious Qwen output; using original consensus'
        else:
            selected, preferred_q = 'q', True
            support = [similarity(q, text) for name, text in original.items()
                       if name != 'q' and text.strip()]
            score = .7 * max(support) + .3 * sum(support) / len(support) if support else 0.0
    if policy == 'nemo':
        r = candidates.get('r', '')
        if compact(r) and not suspicious_text(r):
            selected, preferred_r, preferred_q, score, reason = 'r', True, False, 0.0, None
        else:
            problem = 'empty or punctuation-only' if not compact(r) else 'visibly suspicious'
            reason = (f'Original-audio NeMo output is {problem}; using original Qwen preference'
                      + (f'; {reason}' if reason else ''))
    return dict(selected=selected, initial_selected=selected, score=score,
                preferred_q=preferred_q, preferred_r=preferred_r,
                lock_source=preferred_q or preferred_r, fallback_reason=reason)


def align_with_retry(models, clip, sr, window, candidates, selected, initial, *,
                     lock_source: bool = False):
    """Retry a bad batch item singly, then try independent transcript candidates.

    Never manufacture timestamps for an invalid alignment. A locked preferred
    source requires complete text coverage and can only retry the same source.
    Keep every failure in the audit trail; fail the track if it cannot be aligned.
    """
    failures = []
    try:
        return selected, aligned_tokens(candidates[selected], initial, window,
                                        require_complete=lock_source), failures
    except ValueError as exc:
        failures.append({'model': selected, 'batch': True, 'error': str(exc)})
    alternatives = [] if lock_source else [k for k in ('q', 'w', 'n') if k != selected and candidates.get(k)]
    for name in [selected] + alternatives:
        if name != selected and len(compact(candidates[name])) < .8 * len(compact(candidates[selected])):
            failures.append({'model':name,'batch':False,'error':'Rejected fallback that would discard selected speech'})
            continue
        result = models.align([clip], [candidates[name]], sr)
        if len(result) != 1:
            raise RuntimeError('Aligner retry count mismatch')
        try:
            tokens = aligned_tokens(candidates[name], [asdict(item) for item in result[0].items], window,
                                    require_complete=lock_source)
            logger.warning('Window %d alignment recovered using %s after %d failures',
                           window.index, name, len(failures))
            return name, tokens, failures
        except ValueError as exc:
            failures.append({'model': name, 'batch': False, 'error': str(exc)})
    raise RuntimeError(f'Window {window.index}: no valid forced alignment: {failures}')


def merge_window_utterances(segments: list, evidence: list[dict],
                            utterance_token_rows: list[list[dict]]
                            ) -> tuple[list, list[dict], list[list[dict]]]:
    """Join trailing fragments only at a real change of originating window.

    ``window`` retains the first origin for compatibility. ``last_window`` is
    the immediately preceding origin used by the next boundary decision, while
    ``windows`` records every contributing window in order. Existing grammatical
    and timing eligibility is unchanged.
    """
    merged_segments, merged_evidence, merged_tokens = [], [], []
    import re
    for segment, row, words in zip(segments, evidence, utterance_token_rows):
        # Keep first-origin compatibility, but boundary decisions use the last
        # contributing window after an earlier continuation has been joined.
        row = dict(row, last_window=row['window'], windows=[row['window']])
        previous = merged_segments[-1] if merged_segments else None
        previous_row = merged_evidence[-1] if merged_evidence else None
        continuation = bool(re.match(r'^(?:を|が|の|って|ちゃ|たこと|てしま|なけれ)', segment.text))
        preserve = bool(previous_row and (previous_row.get('source_policy') == 'nemo'
                                          or row.get('source_policy') == 'nemo'))
        # NeMo normally supplies no terminal punctuation, so its absence cannot
        # establish a continuation. Keep the existing explicit suffix signal,
        # with a crop-sized bound; never add punctuation or rewrite raw tokens.
        continuation_allowed = (continuation and segment.end - previous.start <= 30.0
                                if preserve else previous is not None and
                                (not re.search(r'[。！？!?][」』）)]?$', previous.text) or continuation))
        join = (previous is not None and previous_row['last_window'] != row['window']
                and segment.start-previous.end < 2.0 and continuation_allowed)
        if join:
            previous.text = (previous.text if preserve else previous.text.rstrip('。')) + segment.text
            previous.end = segment.end
            previous_row['w'] = previous.text
            for key in ('whisper', 'n', 'q', 'r'):
                if key not in previous_row and key not in row:
                    continue
                prefix = previous_row.get(key, '')
                previous_row[key] = (prefix if preserve else prefix.rstrip('。')) + row.get(key, '')
            previous_row['grade'] = 'C' if 'C' in (previous_row['grade'],row['grade']) else phrase_grade(
                previous.text, {k:previous_row[k] for k in ('whisper','n','q')}, 'hybrid')
            previous_row['source_model'] = ('r' if previous_row['source_model'] == row['source_model'] == 'r'
                                            else 'hybrid')
            previous_row['last_window'] = row['last_window']
            previous_row['windows'].extend(row['windows'])
            previous_row['note'] += f" 与窗口{row['window']}续接为完整句。"
            if merged_tokens[-1] and not preserve:
                merged_tokens[-1][-1] = {**merged_tokens[-1][-1], 'text':merged_tokens[-1][-1]['text'].rstrip('。')}
            merged_tokens[-1].extend(words)
        else:
            segment.index = len(merged_segments)+1
            row['line'] = segment.index
            merged_segments.append(segment); merged_evidence.append(row); merged_tokens.append(list(words))
    return merged_segments, merged_evidence, merged_tokens


def load_track_windows(job: dict, *, maximum: float = 20.0):
    """Extract the same original waveform and independent windows in both passes."""
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    from src.adjudicate import load_full_wav
    from src.audio import extract_audio
    raw = Path(job['audio'])
    raw.parent.mkdir(parents=True, exist_ok=True)
    extract_audio(Path(job['media']), raw)
    sr, audio = load_full_wav(raw)
    speech = get_speech_timestamps(audio, VadOptions(threshold=.35, min_silence_duration_ms=300,
                                                   speech_pad_ms=200))
    windows = audio_windows(len(audio) / sr, speech, sr, maximum=maximum)
    return raw, sr, audio, speech, windows


def audio_window_identity(audio, sr: int, windows) -> dict:
    """Bind complete samples as well as window boundaries, not merely durations."""
    import numpy as np
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('NeMo prepass requires a finite mono waveform')
    return dict(sample_rate=sr, sample_count=len(samples),
                audio_sha256=hashlib.sha256(samples.tobytes(order='C')).hexdigest(),
                windows=[asdict(window) for window in windows])


def preserve_nemo_prepass(path: Path) -> Path | None:
    """Keep every distinct prior attempt before the latest read-only file changes."""
    from src.translate import make_snapshot
    if not path.exists():
        return None
    digest = file_hash(path)
    snapshot = make_snapshot(path, f"attempt-{digest}")
    if file_hash(snapshot) != digest or file_hash(path) != digest:
        raise RuntimeError('NeMo prepass snapshot differs from the prior artifact')
    return snapshot


def prepare_nemo_track(job: dict, spec: dict, path: Path) -> dict:
    """Run the isolated NeMo worker before any resident ensemble GPU models."""
    from src.nemo_asr import NemoASRError, transcribe_windows
    started = time.monotonic()
    preserve_nemo_prepass(path)
    _, sr, audio, _, windows = load_track_windows(job, maximum=spec["config"].get("asr_window_seconds", 20.0))
    identity = audio_window_identity(audio, sr, windows)
    try:
        texts, report = transcribe_windows(
            audio, sr, windows, spec['config'],
            expected_identity=spec.get('nemo_runtime_identity'))
    except NemoASRError as exc:
        write_json(path, dict(input=identity, report=exc.report, complete=False,
                              error=str(exc), seconds=time.monotonic() - started))
        path.chmod(0o444)
        raise
    if len(texts) != len(windows) or any(not isinstance(text, str) for text in texts):
        raise ValueError('NeMo prepass result count/type mismatch')
    payload = dict(input=identity, texts=texts, report=report,
                   seconds=time.monotonic() - started,
                   execution='sequential isolated prepass before W/Q/N model loading')
    write_json(path, payload)
    path.chmod(0o444)
    return dict(path=str(path), sha256=file_hash(path))


def read_nemo_track(binding: dict, audio, sr: int, windows) -> dict:
    path = Path(binding['path'])
    if file_hash(path) != binding['sha256']:
        raise ValueError('NeMo prepass artifact changed')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['input'] != audio_window_identity(audio, sr, windows):
        raise ValueError('NeMo prepass original audio/window identity mismatch')
    if len(payload['texts']) != len(windows) or any(not isinstance(t, str) for t in payload['texts']):
        raise ValueError('NeMo prepass result count/type mismatch')
    return payload


def transcribe_track(job: dict, models: Recognizers, spec: dict, temporary: Path,
                     nemo_prepass: dict | None = None) -> None:
    import numpy as np
    from src.adjudicate import load_full_wav
    from src.audio import separate_vocals
    from src.asr import Segment
    from src.srt import write_srt
    from src.translate import make_snapshot

    started = time.monotonic()
    source = Path(job['source'])
    source.parent.mkdir(parents=True,exist_ok=True)
    raw, sr, audio, speech, windows = load_track_windows(job, maximum=spec["config"].get("asr_window_seconds", 20.0))
    logger.info('Independent original-timeline windows: %d (%.1fs audio)',len(windows),len(audio)/sr)
    source_policy = spec['config'].get('ensemble_source', 'consensus')
    nemo = None
    if source_policy == 'nemo':
        if nemo_prepass is None:
            raise ValueError('NeMo source policy requires a completed isolated prepass')
        nemo = read_nemo_track(nemo_prepass, audio, sr, windows)
    outputs, timings = models.recognize(audio,sr,windows,job['hotwords'])
    if len(outputs) != len(windows):
        raise ValueError('Recognizer window count mismatch')
    if nemo is not None:
        outputs = [dict(row, r=text) for row, text in zip(outputs, nemo['texts'])]
    originals = [dict(row) for row in outputs]
    recovery = []
    for i,row in enumerate(outputs):
        selected, score = choose_transcript({k: v for k, v in row.items() if k in {'w', 'q', 'n'}})
        if any(row.values()) and (score < .82 or suspicious_text(row[selected])):
            recovery.append(i)
    recovery_started = time.monotonic()
    recovery_log = []
    if recovery and not spec['config']['no_demucs']:
        # One separation, then only disputed windows are decoded again. Alternate
        # waveforms remain variants of the same recognizer, not independent votes.
        try:
            vocals = separate_vocals(raw,temporary)
            _, separated = load_full_wav(vocals)
            retries, retry_timings = models.recognize(separated,sr,[windows[i] for i in recovery],[])
            for i,retry in zip(recovery,retries):
                before, old_score = choose_transcript({k: v for k, v in outputs[i].items() if k in {'w', 'q', 'n'}})
                after, new_score = choose_transcript(retry)
                # Require corroboration by a different raw-audio model as well.
                cross = max((similarity(retry[after],v) for k,v in outputs[i].items() if k!=after and k in {'w', 'q', 'n'}),default=0)
                from difflib import SequenceMatcher
                from src.asr_consensus import reading_map
                old_reading = reading_map(outputs[i][before])[0]
                new_reading = reading_map(retry[after])[0]
                matched = sum(m.size for m in SequenceMatcher(None,old_reading,new_reading,autojunk=False).get_matching_blocks())
                retained = matched/max(1,len(old_reading))
                coverage_ok = retained >= .9 or suspicious_text(outputs[i][before])
                eligible = bool(retry[after]) and new_score > old_score + .06 and cross >= .6 and coverage_ok
                adopt = eligible and source_policy not in {'qwen', 'nemo'}
                recovery_log.append(dict(window=i,raw=dict(outputs[i]),vocals=retry,adopted=adopt,
                                         eligible_for_adoption=eligible,
                                         policy_reason='Original-audio source policy: vocal recovery is evidence only' if source_policy in {'qwen', 'nemo'} else None,
                                         raw_score=old_score,vocal_score=new_score,raw_speech_retained=retained))
                if adopt:
                    outputs[i] = retry
            timings['recovery_recognizers'] = retry_timings
        except (RuntimeError,OSError) as exc:
            logger.warning('Vocal recovery unavailable; retaining raw evidence: %s',exc)
            recovery_log.append({'error':str(exc)})
    recovery_seconds = time.monotonic()-recovery_started
    source_selection = [dict(window=i, **select_window_source(row, source_policy)) for i, row in enumerate(outputs)]
    selections = [(row['selected'], row['score']) for row in source_selection]
    for i, (selected, _) in enumerate(selections):
        if not source_selection[i]['lock_source'] and source_policy != 'nemo':
            outputs[i][selected] = restore_sentence_boundaries(outputs[i][selected], outputs[i].get('q', ''))
    wanted = [i for i,(name,_) in enumerate(selections) if outputs[i][name].strip() and np.any(audio[int(windows[i].start*sr):int(windows[i].end*sr)])]
    segments, evidence, aligned = [], [], {}
    utterance_token_rows = []
    source_unit_splits = []
    alignment_retries = []
    align_started = time.monotonic()
    for offset in range(0,len(wanted),models.batch_size):
        ids = wanted[offset:offset+models.batch_size]
        results = models.align([audio[int(windows[i].start*sr):int(windows[i].end*sr)] for i in ids],
                               [outputs[i][selections[i][0]] for i in ids],sr)
        if len(results)!=len(ids):
            raise RuntimeError('Aligner batch count mismatch')
        for i,result in zip(ids,results):
            selected, aligned[i], failures = align_with_retry(
                models, audio[int(windows[i].start*sr):int(windows[i].end*sr)], sr,
                windows[i], outputs[i], selections[i][0], [asdict(item) for item in result.items],
                lock_source=source_selection[i]['lock_source'])
            selections[i] = (selected, selections[i][1])
            source_selection[i]['selected'] = selected
            if failures:
                alignment_retries.append(dict(window=i, selected=selected, failures=failures))
        logger.info('Aligned independent windows %d/%d',min(offset+models.batch_size,len(wanted)),len(wanted))
    align_seconds = time.monotonic()-align_started
    previous_end = 0.0
    for i in wanted:
        selected, score = selections[i]
        full = outputs[i][selected]
        tokens = aligned[i]
        # Ownership removes most overlap duplicates. Alignments can jitter at the
        # cut, so remove only matching tokens whose times actually overlap.
        if segments and tokens and tokens[0]['start'] < previous_end:
            while tokens and tokens[0]['end'] <= previous_end and compact(segments[-1].text).endswith(compact(tokens[0]['text'])):
                tokens.pop(0)
        grouped_tokens = []
        for original_group in utterance_groups(tokens):
            if spec['config'].get('source_pause_units', False):
                from src.source_units import pause_token_groups
                parts, cuts = pause_token_groups(original_group, speech, sr)
                if cuts:
                    source_unit_splits.append(dict(
                        window=i, original_text=''.join(t['text'] for t in original_group).strip(),
                        cuts=cuts, part_count=len(parts),
                        evidence_scope='Timing-supported translation-unit hint, not a speaker or source-correctness decision'))
                grouped_tokens.extend(parts)
            else:
                grouped_tokens.append(original_group)
        for group in grouped_tokens:
            text = ''.join(t['text'] for t in group).strip()
            if not text:
                continue
            alternatives = {k:project_phrase(full,v,group[0]['begin'],group[-1]['finish'])
                            for k,v in outputs[i].items()}
            alternatives[selected] = text
            grade = ('C' if source_policy == 'nemo' else phrase_grade(text,alternatives,selected))
            start,end = group[0]['start'],group[-1]['end']
            alignment_problem = end-start < .08 or start < previous_end-.15 or end-start > 8
            start = max(start,previous_end)
            end = min(len(audio)/sr,max(end,start+.08))
            if end <= start:
                raise RuntimeError('Unable to construct a positive subtitle interval')
            retried_alignment = any(row['window'] == i for row in alignment_retries)
            if alignment_problem or retried_alignment or suspicious_text(text):
                grade='C'
            index=len(segments)+1
            segments.append(Segment(index,start,end,text))
            utterance_token_rows.append(group)
            evidence.append(dict(line=index,w=text,whisper=alternatives.get('w',''),
                                 n=alternatives.get('n',''),q=alternatives.get('q',''),grade=grade,
                                 **({'r': alternatives.get('r', ''), 'source_policy': 'nemo'}
                                    if source_policy == 'nemo' else {}),
                                 source_model=selected,ensemble=True,window=i,
                                 note=f'独立原音频窗口；选用{selected.upper()}；逐短语声学证据。'
                                      + (' NeMo仅为暂选源文，与N同属Reazon系，不增加独立票数；需复听。' if source_policy == 'nemo' else '')
                                      + (' 对齐异常或重试，需复听。' if alignment_problem or retried_alignment else '')))
            previous_end=end
    segments, evidence, utterance_token_rows = merge_window_utterances(
        segments, evidence, utterance_token_rows)
    metadata=dict(version=spec['version'],strategy='ensemble',windows=[asdict(w) for w in windows],
                  asr_window_seconds=spec['config'].get('asr_window_seconds', 20.0),
                  ensemble_source=source_policy, qwen_asr_hotwords=spec['config'].get('qwen_asr_hotwords', True),
                  source_pause_units=spec['config'].get('source_pause_units', False),
                  source_unit_splits=source_unit_splits,
                  source_selection=source_selection,
                  qwen_text_provenance='Original-audio hypotheses are SDK-processed; repetition cleanup can hide decoder runaways',
                  raw_transcripts=originals,used_transcripts=outputs,recovery=recovery_log,
                  recognizer_timings=timings,recovery_seconds=recovery_seconds,
                  alignment_seconds=align_seconds,alignment_retries=alignment_retries,
                  utterance_tokens=utterance_token_rows, speech_regions=speech, sample_rate=sr,
                  segments=[asdict(s) for s in segments])
    if nemo is not None:
        metadata['nemo_prepass'] = nemo
        metadata['nemo_independent_vote'] = False
    prepass_seconds = nemo['seconds'] if nemo is not None else 0.0
    state=StageState(Path(job['state']))
    if source.exists():make_snapshot(source,'pre-asr')
    if not segments:
        source.unlink(missing_ok=True)
        state.save('asr',job['key'],[],status='empty',seconds=time.monotonic()-started+prepass_seconds)
        return
    with tempfile.TemporaryDirectory(prefix='.ensemble-',dir=source.parent) as directory:
        staged=Path(directory)/source.name
        write_srt(segments,staged)
        staged.replace(source)
    meta=Path(job.get('metadata') or artifact_path(source,'.asr.json'))
    adjudication=Path(job.get('adjudication') or artifact_path(source,'.adjudication.json'))
    write_json(meta,metadata)
    write_json(adjudication,evidence)
    total_seconds = time.monotonic() - started + prepass_seconds
    artifacts = [source, meta] + ([Path(nemo_prepass['path'])] if nemo_prepass else [])
    state.save('asr',job['key'],artifacts,seconds=total_seconds,strategy='ensemble')
    key=fingerprint([spec['version'],job['media_hash'],file_hash(source),spec['source_lang'],spec['batch_size']])
    state.save('arbitration',key,[adjudication],seconds=0.0,included_in='asr')
    logger.info('Ensemble complete: %d rebuilt cues, %d recovery windows, %.2fs',len(segments),len(recovery),total_seconds)


def run(spec: dict) -> int:
    failed, pending = 0, []
    for job in spec['jobs']:
        binding = None
        if spec['config'].get('ensemble_source') == 'nemo':
            path = artifact_path(Path(job['source']), '.nemo-prepass.json')
            try:
                binding = prepare_nemo_track(job, spec, path)
            except Exception as exc:
                logger.exception('NeMo prepass failed for %s', job['media'])
                StageState(Path(job['state'])).save('asr', job['key'], [], status='failed', error=str(exc))
                failed += 1
                continue
        pending.append((job, binding))
    if not pending:
        return int(failed > 0)
    models = Recognizers(spec['config'], spec['batch_size'])
    for job, binding in pending:
        try:
            with tempfile.TemporaryDirectory(prefix='autosub-ensemble-') as temporary:
                transcribe_track(job, models, spec, Path(temporary), nemo_prepass=binding)
        except Exception as exc:
            logger.exception('Ensemble failed for %s',job['media'])
            StageState(Path(job['state'])).save('asr',job['key'],[],status='failed',error=str(exc))
            failed += 1
    return int(failed > 0)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
    sys.exit(run(json.loads(Path(sys.argv[1]).read_text())))
