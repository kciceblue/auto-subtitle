"""Optional local Chinese editing; every returned line remains unverified.

Only the raw-response cache is written here. Callers commit the returned targets
only after the entire stage succeeds, preserve source uncertainties, and obtain
an independent score separately. Evaluator findings are never writer inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit

from src.config import TranslateConfig
from src.quality import validate_blocks
from src.translate import SrtBlock, call_llm, read_context_file
from src.workflow_state import fingerprint, write_json

VERSION = 'coherence-editor-2'
MAX_CONTEXT_CHARS = 32000
MAX_DRAFT_CHARS = 32000
EDIT = '''你是中文字幕编辑。根据原文核对本次待编辑译文，使其在连续阅读时自然、清楚、逻辑通顺。
全片中文草稿只是未经校验的上下文，不是事实依据。原文可能来自有错误的语音识别。
只修正文法、搭配、明显别扭的直译和有上下文依据的指代；可以调整语序。
保留每条原文实际表达的内容、否定、人物关系、数字和不确定性，不得靠编造情节来使其通顺。
正常的口语省略、感叹和短回答不需要补写成完整句子。不可跨ID挪动、删除或合并台词。
无法由给出的原文和上下文支持的改动不要做。人名译法保持一致，不添加未出现的说话人。
输出JSON对象，键为本次连续编号，值为编辑后的完整中文。每个编号必须齐全；无需改动则照抄。
禁止输出说明、注释、Markdown或额外编号。
'''
_TIMESTAMP = re.compile(r'\d{2}:\d{2}:\d{2}[,.]\d{3}')
_THINKING = re.compile(r'</?(?:think(?:ing)?|analysis|reasoning)\b|<\|[^>]*\|>', re.I)


class CoherenceError(RuntimeError):
    """No complete editor result is available; prior outputs must remain intact."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f'Invalid JSON constant: {value}')


def _json(text: str):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _edited_text(text: object, source: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty or non-string editor output')
    if (re.search(r'\n\s*\n', text) or _TIMESTAMP.search(text) or _THINKING.search(text)
            or any(marker in text for marker in ('```', '⟦', '⟧'))
            or any(unicodedata.category(char) in {'Cc', 'Cf', 'Cs'} and char != '\n'
                   for char in text)):
        raise ValueError('Editor output contains forbidden formatting, control characters or reasoning')
    value = '\n'.join(line.strip() for line in text.strip().splitlines())
    if len(value) > max(160, 4 * len(source)):
        raise ValueError('Runaway editor output')
    return value


def _parse_response(response: str, source: list[SrtBlock]) -> list[str]:
    if not isinstance(response, str):
        raise ValueError('Editor response must be a JSON string')
    parsed = _json(response)
    expected = {str(i) for i in range(1, len(source) + 1)}
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ValueError('Editor response must contain exactly the requested local IDs')
    return [_edited_text(parsed[str(i)], row.text) for i, row in enumerate(source, 1)]


def _original_text(path: Path, raw: bytes) -> str:
    if path.suffix.lower() == '.pdf':
        # The general reader skips individual failed PDF pages. This stage must
        # retain complete context or fail, so never accept a partial extraction.
        try:
            from pypdf import PdfReader
            text = '\n'.join((page.extract_text() or '').strip()
                             for page in PdfReader(BytesIO(raw)).pages).strip()
        except Exception as exc:
            raise ValueError(f'Original PDF context could not be read completely: {path}') from exc
    else:
        text = read_context_file(path)
        if path.read_bytes() != raw:
            raise ValueError(f'Original context changed while reading: {path}')
    if raw.strip() and not text:
        raise ValueError(f'Original context could not be read completely: {path}')
    return text


def _context(config: TranslateConfig) -> tuple[str, dict]:
    # Deliberately exclude line/entity notes, adjudication and reference subtitles.
    # Files take precedence over a potentially lossy context summary.
    parts, evidence = [], {'files': [], 'title': config.title, 'summary': None, 'vocab': None}
    if config.title:
        parts.append('媒体标题：\n' + config.title)
    for path in config.context_files:
        path = Path(path)
        raw = path.read_bytes()
        text = _original_text(path, raw)
        evidence['files'].append({'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'text': text})
        parts.append(f'原始资料 {path.name}：\n{text}')
    if not config.context_files and config.context_summary:
        evidence['summary'] = config.context_summary
        parts.append(config.context_summary)
    if config.vocab_file is not None:
        path = Path(config.vocab_file)
        raw = path.read_bytes()
        text = _original_text(path, raw)
        evidence['vocab'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'text': text}
        parts.append('原始词表（仅词语背景，不可覆盖原文）：\n' + text)
    context = '\n\n'.join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        raise ValueError('Complete original context exceeds the coherence guard; refusing truncation')
    return context, evidence


def _request_config(config: TranslateConfig, source: list[SrtBlock]) -> TranslateConfig:
    if config.target_lang_code.lower().split('-')[0] != 'zh':
        raise ValueError('Coherence editing currently supports Chinese targets only')
    parsed = urlsplit(config.endpoint)
    if (parsed.scheme not in {'http', 'https'}
            or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username is not None or parsed.password is not None):
        raise ValueError('Coherence editing requires a local loopback endpoint')
    if config.extra_payload is not None and not isinstance(config.extra_payload, dict):
        raise ValueError('Editor extra_payload must be an object')
    extra = deepcopy(config.extra_payload or {})
    model = extra.get('model')
    if (not isinstance(model, str) or not model.strip()
            or model.strip().casefold() in {'gpt-6-astra', 'fable-5.1'}):
        raise ValueError('Identify the actual local editor model; external evaluators cannot edit')
    if any(key in extra for key in ('messages', 'prompt', 'system', 'input', 'stream', 'stream_options')):
        raise ValueError('Editor payload cannot override the bound source and draft input')
    if any(key in extra for key in ('grammar', 'json_schema')):
        raise ValueError('Editor requires its own exact-ID response schema')
    max_tokens = extra.get('max_tokens', config.max_tokens)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('Editor max_tokens must be a positive integer')
    kwargs = deepcopy(extra.get('chat_template_kwargs') or {})
    if not isinstance(kwargs, dict):
        raise ValueError('Editor chat_template_kwargs must be an object')
    kwargs['enable_thinking'] = False
    for key in ('reasoning_budget_tokens', 'thinking_budget_tokens', 'reasoning_budget',
                'reasoning_budget_message'):
        extra.pop(key, None)
    extra.update(reasoning_effort='none', chat_template_kwargs=kwargs, max_tokens=max_tokens,
                 response_format={'type': 'json_object', 'schema': {
                     'type': 'object', 'properties': {
                         str(i): {'type': 'string', 'minLength': 1,
                                  'maxLength': max(160, 4 * len(row.text))}
                         for i, row in enumerate(source, 1)},
                     'required': [str(i) for i in range(1, len(source) + 1)],
                     'additionalProperties': False}})
    # All sampler/model settings survive; only this stage's format/thinking policy changes.
    json.dumps(extra, allow_nan=False)
    return replace(config, extra_payload=extra, max_tokens=max_tokens, retries=0,
                   translation_reasoning_budget=0, stage='coherence_edit',
                   response_guard_floor=max(2048, config.response_guard_floor))


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {'version': VERSION, 'runs': {}}
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8')
        saved = _json(text)
        # Reject invalid Unicode too, before an eventual cache write could fail.
        json.dumps(saved, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if (isinstance(saved, dict) and saved.get('version') == VERSION
                and isinstance(saved.get('runs'), dict)):
            return saved
    except (UnicodeError, ValueError, TypeError):
        pass
    # Preserve a malformed/older cache for inspection rather than erasing its bytes.
    return {'version': VERSION, 'runs': {}, 'previous_cache_hex': raw.hex()}


def polish_coherence(source: list[SrtBlock], draft: list[SrtBlock], config: TranslateConfig,
                     cache: Path, *, batch_size: int = 20) -> tuple[list[SrtBlock], list[dict]]:
    """Edit complete matched targets locally; raise if any batch remains unchecked.

    At most two calls to the shared request helper are made for each failed batch;
    that helper's transport retries are disabled (its HTTP-400 streaming fallback
    remains shared behavior). No earlier edited batch becomes later context.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Coherence batch_size must be a positive integer')
    if (not isinstance(source, list) or not isinstance(draft, list)
            or any(not isinstance(row, SrtBlock) or not isinstance(row.text, str)
                   or not isinstance(row.ts_line, str) for row in [*source, *draft])):
        raise ValueError('Source and draft must contain subtitle blocks with text and timestamps')
    source, draft = deepcopy(source), deepcopy(draft)
    validate_blocks(source)
    validate_blocks(draft)
    if (len(source) != len(draft)
            or any(type(a.index) is not int or type(b.index) is not int
                   or a.index != b.index or a.ts_line != b.ts_line
                   for a, b in zip(source, draft))):
        raise ValueError('Source and draft IDs, counts and timestamps must match exactly')
    cache = Path(cache)
    protected = [p for p in (config.input_srt, config.output_srt, config.vocab_file,
                            config.reference_srt, config.script_file, *config.context_files) if p is not None]
    if cache.suffix != '.json' or cache.resolve() in {Path(p).resolve() for p in protected}:
        raise ValueError('Use a distinct coherence JSON cache, never an input or subtitle file')
    context, context_evidence = _context(config)
    whole = '\n'.join(row.text for row in draft)
    if len(whole) > MAX_DRAFT_CHARS:
        raise ValueError('Complete Chinese draft exceeds the coherence guard; refusing truncation')
    instruction = EDIT + '\n作品背景：\n' + context + '\n全片中文草稿（只读）：\n' + whole
    first_config = _request_config(config, source[:batch_size])
    binding = {'version': VERSION, 'source': [asdict(row) for row in source],
               'draft': [asdict(row) for row in draft], 'context': context_evidence,
               'instruction': instruction, 'batch_size': batch_size,
               'endpoint': first_config.endpoint, 'settings': first_config.extra_payload,
               'max_tokens': first_config.max_tokens, 'timeout': first_config.timeout,
               'transport_retries': 0, 'with_thinking': False,
               'stream_usage_requested': first_config.telemetry_path is not None,
               'separate_instruction': first_config.separate_instruction}
    run_key = fingerprint(binding)
    saved = _load_cache(cache)
    run = saved['runs'].get(run_key)
    if not isinstance(run, dict) or run.get('binding') != binding or not isinstance(run.get('batches'), dict):
        run = {'binding': binding, 'batches': {}, 'status': 'UNVERIFIED'}
        saved['runs'][run_key] = run
    final, ledger = [], []
    for start in range(0, len(source), batch_size):
        src, old = source[start:start + batch_size], draft[start:start + batch_size]
        cfg = _request_config(config, src)
        body = json.dumps({str(i): {'source': a.text, 'chinese': b.text}
                           for i, (a, b) in enumerate(zip(src, old), 1)}, ensure_ascii=False)
        request = {'run_key': run_key, 'global_ids': [row.index for row in src], 'body': body,
                   'endpoint': cfg.endpoint, 'settings': cfg.extra_payload,
                   'max_tokens': cfg.max_tokens, 'timeout': cfg.timeout,
                   'response_guard_floor': cfg.response_guard_floor,
                   'transport_retries': cfg.retries, 'with_thinking': False,
                   'stream_usage_requested': cfg.telemetry_path is not None,
                   'separate_instruction': cfg.separate_instruction}
        key = fingerprint(request)
        entry = run['batches'].get(key)
        if not isinstance(entry, dict) or entry.get('request') != request:
            entry = {'request': request, 'attempts': [], 'status': 'UNVERIFIED'}
            run['batches'][key] = entry
        hit, values = False, None
        if not isinstance(entry.get('cache_rejections'), list):
            entry['cache_rejections'] = []
        response = entry.get('response')
        if isinstance(response, str):
            try:
                if entry.get('response_sha256') != _text_hash(response):
                    raise ValueError('Cached editor response checksum differs')
                values = _parse_response(response, src)
                hit = True
            except (ValueError, TypeError, UnicodeError) as exc:
                rejected = {'response': response, 'error': str(exc)}
                entry['cache_rejections'].append(rejected)
                entry.pop('response', None)
                entry.pop('response_sha256', None)
        elif response is not None:
            entry['cache_rejections'].append({'response': response, 'error': 'Cached response is not a string'})
            entry.pop('response', None)
            entry.pop('response_sha256', None)
        if not isinstance(entry.get('attempts'), list):
            entry['attempts'] = []
        if values is None:
            for attempt in range(2):
                began = time.monotonic()
                response, error = None, None
                try:
                    response = call_llm(body, instruction, cfg)
                    values = _parse_response(response, src)
                except (ValueError, TypeError, UnicodeError, RuntimeError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                entry['attempts'].append({'attempt': attempt + 1, 'response': response,
                                          'error': error, 'seconds': time.monotonic() - began})
                if error is None:
                    entry.update(response=response, response_sha256=_text_hash(response))
                write_json(cache, saved)
                if error is None:
                    break
            if values is None:
                raise CoherenceError(f'Coherence batch {start + 1} remained unchecked after two attempts; '
                                     f'no target result committed. See {cache}')
        for a, b, text in zip(src, old, values):
            final.append(replace(b, text=text))
            ledger.append({'line': a.index, 'before': b.text, 'after': text,
                           'changed': b.text != text, 'model': cfg.extra_payload['model'],
                           'cached': hit, 'status': 'UNVERIFIED', 'request_key': key,
                           'source_sha256': _text_hash(a.text), 'draft_sha256': _text_hash(b.text),
                           'context_sha256': fingerprint(context_evidence),
                           'response_sha256': entry['response_sha256'],
                           'source_uncertainty_cleared': False})
    validate_blocks(final)
    if len(final) != len(source) or any(a.index != b.index or a.ts_line != b.ts_line
                                      for a, b in zip(source, final)):
        raise CoherenceError('Editor changed source/target structural alignment')
    return final, ledger
