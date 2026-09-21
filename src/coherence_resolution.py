"""Optional Chinese-only local repair with explicit issue-resolution receipts.

This module owns neither backend lifetime nor subtitle files. The caller freezes
inputs, owns the local backend, commits complete returned candidates, and obtains
an independent score. A local resolved state is never release verification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from urllib.parse import urlsplit

from src.coherence import (CoherenceError, MAX_DRAFT_CHARS, _context, _edited_text,
                           _json, _load_cache, _parse_response, _request_config)
from src.coherence_refine import _diagnosis_config, _query
from src.config import TranslateConfig
from src.local_backend import _request_json
from src.quality import validate_blocks
from src.translate import SrtBlock, _build_payload, _THINKING_OFF_VARIANTS
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'local-repair-resolution-2'
DIAGNOSIS_POLICIES = ('conservative', 'recall-precheck')
PRECHECK_STATUSES = ('persists', 'diagnosis_unsupported')
STATUSES = ('resolved', 'persists', 'diagnosis_unsupported')
DIAGNOSE = '''你是本地中文对白检查员，只检查中文本身显而易见的逻辑、措辞、指代和问答问题，不判断日语忠实度。正常省略、短答、诗歌、换说话人和场景转换不算错误，不确定的猜测不要报。
全片中文带稳定global_id和时间戳，只供定位和理解；本轮items是要检查的条目。每个本地编号必须输出status(OK或ISSUE)、reason和context_ids。ISSUE需要简短具体的中文依据；context_ids只能引用给出的全局编号，是只读相关语境，不是待改行。OK的reason为空且context_ids为空。不要给出修正文案。'''
REPAIR = '''你是本地中文对白编辑。只根据当前中文、所给作品背景和本地检查意见，修复items里每个问题。每个问题只允许改它自己的global_id；相关context_ids和全片其他条目均为只读。保持每条实际传达的信息、否定、数字和不确定性；不得移走、合并、删掉台词或编造事件来消除矛盾。可以依据完整中文中明确的证据修正指代或统一已有名字变体，证据不足不要猜。
检查意见也可能错，若原句无该问题可原样返回；不要为了有变化而改写。previous_resolution是本地上次检查结果；persists时认真解决它，而不是重复无效改法。输出每个本地编号对应的完整中文字符串，不输出分析或说明。'''
RESOLVE = '''你是本地中文对白复核员。你会看到本地最初的检查意见、原来的被检查句、修订后的当前句，以及完整的当前中文字幕。独立判断每个原问题在当前完整文本中是否仍然成立，不要因为编辑模型做了修改就认定修好了；原样未改的句子同样必须检查。
每个本地编号必须给出status和简短具体reason：resolved表示原诊断合理且当前文本已消除该问题；persists表示当前仍有该问题；diagnosis_unsupported表示原诊断本身没有足够中文依据，应撤回而非强迫改写。不要给替换措辞。只判断可从中文看出的明显问题；正常口语省略、诗歌、换说话人或场景变化不是错误。结论只是本地未验证意见，不是发布批准，也不表示日语忠实。'''


RECALL_DIAGNOSE = """你是本地中文对白问题候选检查员，只阅读中文本身和给出的中文上下文，不判断日语忠实度。逐条寻找值得进一步核查的具体措辞、逻辑、指代或问答问题；当问题有明确的文字依据但仍需核查时，可以报告候选，不必先自行证明无疑。
每个候选必须指出当前中文中具体可能不成立的表达或关系，context_ids只能引用给出的相关全局编号。不能凭未知剧情、缺少视频、主观风格偏好或抽象的“不自然”凑问题；没有固定候选数量要求。正常口语省略、短答、诗歌、说话人更换和场景变化本身不是问题。
本轮只收集候选，不授权修改。每个本地编号都必须输出status(OK或ISSUE)、简短具体的reason和只读context_ids；OK的reason和context_ids为空。不要给替换措辞、补写剧情或报告本轮items之外的编号。"""
PRECHECK = """你是本地中文对白候选复核员。尚未进行任何修改：original_chinese和current_chinese来自同一份完整当前草稿。每条候选只是待核查的本地意见，不能因它被提出就认定有问题。
结合完整中文判断候选是否指出了当前确实存在的明显措辞、逻辑、指代或问答问题。每个本地编号必须输出status和简短具体reason。只能选persists或diagnosis_unsupported：persists表示从当前中文可以确立该问题，需要修复；diagnosis_unsupported表示证据不足、诊断不成立或只是允许的表达变化，应撤回并保留原文。原文未修改，不允许resolved状态。
正常口语省略、短答、诗歌、说话人更换和场景变化不算问题；不要猜测日语忠实度、未知视频或剧情。不输出替换措辞，不强迫改写，不新增问题编号。这只是本地未验证的筛查，不是发布批准。"""


def _whole(rows: list[SrtBlock]) -> str:
    if len('\n'.join(row.text for row in rows)) > MAX_DRAFT_CHARS:
        raise ValueError('Complete current Chinese context exceeds guard; no truncation allowed')
    return json.dumps({str(row.index): {'global_id': row.index, 'timestamp': row.ts_line,
                                      'chinese': row.text} for row in rows}, ensure_ascii=False)


def _record_schema(rows: list[SrtBlock], item: dict) -> dict:
    return {'type': 'object', 'properties': {str(i): deepcopy(item) for i in range(1, len(rows) + 1)},
            'required': [str(i) for i in range(1, len(rows) + 1)], 'additionalProperties': False}


def _records(raw: str, count: int) -> dict:
    data = _json(raw)
    if not isinstance(data, dict) or set(data) != {str(i) for i in range(1, count + 1)}:
        raise ValueError('Every requested issue or cue requires exactly one local-ID result')
    return data


def _reason(value: object, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 240:
        raise ValueError('Local reason must be a string of at most240 characters')
    if empty:
        if value != '':
            raise ValueError('OK requires an empty reason')
        return ''
    return _edited_text(value, 'x' * 60)


def parse_diagnosis(raw: str, rows: list[SrtBlock], all_ids: set[int]) -> dict[int, dict]:
    data, result = _records(raw, len(rows)), {}
    for i, row in enumerate(rows, 1):
        item = data[str(i)]
        if (not isinstance(item, dict) or set(item) != {'status', 'reason', 'context_ids'}
                or item['status'] not in {'OK', 'ISSUE'} or not isinstance(item['context_ids'], list)
                or len(item['context_ids']) > 8
                or any(type(ref) is not int or ref not in all_ids for ref in item['context_ids'])):
            raise ValueError('Invalid local diagnosis or unknown context cue')
        reason = _reason(item['reason'], empty=item['status'] == 'OK')
        if item['status'] == 'OK' and item['context_ids']:
            raise ValueError('OK requires no related problem cues')
        result[row.index] = {'status': item['status'], 'reason': reason,
                             'context_ids': list(dict.fromkeys(item['context_ids']))}
    return result


def parse_resolution(raw: str, rows: list[SrtBlock]) -> dict[int, dict]:
    data, result = _records(raw, len(rows)), {}
    for i, row in enumerate(rows, 1):
        item = data[str(i)]
        if (not isinstance(item, dict) or set(item) != {'status', 'reason'}
                or item['status'] not in STATUSES):
            raise ValueError('Invalid local resolution status or fields')
        result[row.index] = {'status': item['status'], 'reason': _reason(item['reason'])}
    return result


def parse_precheck(raw: str, rows: list[SrtBlock]) -> dict[int, dict]:
    result = parse_resolution(raw, rows)
    if any(item['status'] not in PRECHECK_STATUSES for item in result.values()):
        raise ValueError('Unchanged-draft precheck permits only persists or diagnosis_unsupported')
    return result


def capacity_check(body: str, instruction: str, config: TranslateConfig, *,
                   context_size: int, thinking: bool) -> dict:
    """Count the exact native rendered request, reserving the entire output budget."""
    url = urlsplit(config.endpoint)
    if (url.scheme not in {'http', 'https'} or url.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or url.username or url.password or url.query or url.fragment):
        raise ValueError('Capacity preflight requires a plain local loopback endpoint')
    def endpoint(path):
        return url._replace(path=path).geturl()
    props = _request_json(endpoint('/props'), timeout=30)
    observed = props.get('default_generation_settings', {}).get('n_ctx') if isinstance(props, dict) else None
    if type(observed) is not int or observed < context_size:
        raise ValueError('Native backend did not confirm the required context capacity')
    payload = _build_payload(body, instruction, config, _THINKING_OFF_VARIANTS[0], config.max_tokens,
                             stream=True, with_thinking=thinking)
    rendered = _request_json(endpoint('/apply-template'), payload, timeout=30)
    prompt = rendered.get('prompt') if isinstance(rendered, dict) else None
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('Native template returned no prompt')
    tokens = _request_json(endpoint('/tokenize'), {'content': prompt, 'add_special': True,
                                                  'parse_special': True}, timeout=30)
    tokens = tokens.get('tokens') if isinstance(tokens, dict) else None
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError('Native token preflight returned invalid tokens')
    return {'prompt_sha256': fingerprint(prompt), 'request_sha256': fingerprint(payload),
            'prompt_tokens': len(tokens), 'reserved_completion_tokens': config.max_tokens,
            'safety_margin_tokens': 64, 'context_size': context_size, 'backend_context_size': observed,
            'fits': len(tokens) + config.max_tokens + 64 <= context_size,
            'full_context_preserved': True, 'generation_performed': False}


def refine_resolution(source: list[SrtBlock], draft: list[SrtBlock], config: TranslateConfig,
                      cache: Path, *, batch_size: int = 20, critic_budget: int = 1536,
                      edit_budget: int = 1536, max_repair_retries: int = 2,
                      context_size: int = 32768, prior_score: int | None = None,
                      diagnosis_policy: str = 'conservative'
                      ) -> tuple[list[SrtBlock], list[dict]]:
    """Return an unverified complete candidate; unresolved issues remain explicit.

    Each diagnosed issue owns one cue. Every cycle checks all original issues
    against the complete assembled result, even if their text did not change or
    they were previously resolved/withdrawn. Only persisting issues are retried.
    Recall-precheck first verifies every candidate on the unchanged draft; only
    persisting candidates can be edited. Unsupported candidates are still
    rechecked after later edits and can reopen within the same global cycle cap.
    A malformed/incomplete response raises without returning a partial candidate.
    """
    if not isinstance(diagnosis_policy, str) or diagnosis_policy not in DIAGNOSIS_POLICIES:
        raise ValueError('Unknown local diagnosis policy')
    for name, value, low, high in [('batch_size', batch_size, 1, 40),
            ('critic_budget', critic_budget, 0, 4096), ('edit_budget', edit_budget, 0, 4096),
            ('max_repair_retries', max_repair_retries, 0, 2), ('context_size', context_size, 4096, 131072)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Invalid repair-resolution ' + name)
    if prior_score is not None and (type(prior_score) is not int or not -10 <= prior_score <= 4):
        raise ValueError('Only an aggregate numeric prior score is accepted')
    if (not isinstance(source, list) or not isinstance(draft, list)
            or any(not isinstance(row, SrtBlock) or type(row.index) is not int
                   or not isinstance(row.text, str) or not isinstance(row.ts_line, str)
                   for row in [*source, *draft])):
        raise ValueError('Source and draft must contain subtitle blocks')
    source, draft = deepcopy(source), deepcopy(draft)
    validate_blocks(source); validate_blocks(draft)
    if len(source) != len(draft) or any(a.index != b.index or a.ts_line != b.ts_line for a, b in zip(source, draft)):
        raise ValueError('Source and draft IDs, counts and timestamps must match exactly')
    config = replace(config, separate_instruction=True)
    base = _request_config(config, draft[:batch_size])
    if not 512 <= base.max_tokens <= 8192 or max(critic_budget, edit_budget) >= base.max_tokens:
        raise ValueError('Use512..8192 output tokens with room after the reasoning budget')
    if not 0 < config.timeout <= 600:
        raise ValueError('Use a bounded positive local request timeout at most600 seconds')
    cache = Path(cache)
    protected = [p for p in (config.input_srt, config.output_srt, config.vocab_file,
                             config.reference_srt, config.script_file, *config.context_files) if p is not None]
    if cache.suffix != '.json' or cache.resolve() in {Path(p).resolve() for p in protected}:
        raise ValueError('Use a distinct local resolution JSON cache')
    context, context_evidence = _context(config)
    _whole(draft)
    binding = {'version': VERSION, 'code_sha256': file_hash(Path(__file__)),
        'source': [asdict(row) for row in source], 'draft': [asdict(row) for row in draft],
        'context': context_evidence, 'diagnosis_policy': diagnosis_policy,
        'prompts': {'diagnosis': DIAGNOSE, 'repair': REPAIR, 'resolution': RESOLVE,
                    'recall_diagnosis': RECALL_DIAGNOSE, 'precheck': PRECHECK},
        'batch_size': batch_size, 'critic_budget': critic_budget, 'edit_budget': edit_budget,
        'max_repair_retries': max_repair_retries, 'context_size': context_size, 'prior_score': prior_score,
        'endpoint': base.endpoint, 'settings': base.extra_payload, 'timeout': base.timeout,
        'separate_instruction': True, 'max_tokens': base.max_tokens}
    run_key = fingerprint(binding); saved = _load_cache(cache)
    run = saved['runs'].get(run_key)
    if not isinstance(run, dict) or run.get('binding') != binding or not isinstance(run.get('batches'), dict):
        run = {'binding': binding, 'batches': {}, 'status': 'UNVERIFIED'}
        saved['runs'][run_key] = run
    run['complete'] = False
    final = deepcopy(draft); all_ids = {row.index for row in draft}; diagnostics = {}; receipts = {}
    histories = {row.index: [] for row in draft}; attempts = {row.index: 0 for row in draft}
    previous = {}; original = {row.index: row for row in draft}
    initial_prechecks, precheck_receipts = {}, {}
    seed = base.extra_payload.get('seed', 0)
    if type(seed) is not int:
        raise ValueError('Local repair seed must be an integer')

    def query(phase, rows, body, prompt, snapshot, schema, budget, cycle, parser):
        local = replace(config, extra_payload={**(config.extra_payload or {}), 'seed': seed + cycle})
        cfg = _diagnosis_config(local, rows, schema, budget, 'local_resolution_' + phase)
        instruction = prompt + '\n作品背景（只读）：\n' + context + '\n完整当前中文（只读global_id）：\n' + _whole(snapshot)
        if prior_score is not None:
            instruction += '\n唯一外部反馈是整体数字评分' + str(prior_score) + '，目标4；没有外部问题位置或建议。'
        request_body = {'cycle': cycle, 'current_draft_sha256': fingerprint([asdict(r) for r in snapshot]),
                        'items': body}
        if diagnosis_policy == 'recall-precheck':
            request_body['diagnosis_policy'] = diagnosis_policy
        body = json.dumps(request_body, ensure_ascii=False)
        preflight = capacity_check(body, instruction, cfg, context_size=context_size, thinking=bool(budget))
        bound = fingerprint({'phase': phase, 'body': body, 'instruction': instruction,
                             'settings': cfg.extra_payload, 'max_tokens': cfg.max_tokens})
        run.setdefault('preflights', {})[bound] = preflight
        write_json(cache, saved)
        if not preflight['fits']:
            raise CoherenceError('Complete local resolution request exceeds native context capacity; no truncation')
        parsed, receipt = _query(cache, saved, run, run_key, body, instruction, cfg, [row.index for row in rows],
                                 parser, thinking=bool(budget))
        return parsed, {**receipt, 'model': cfg.extra_payload['model'], 'diagnosis_policy': diagnosis_policy,
                        'settings': deepcopy(cfg.extra_payload), 'stage': phase,
                        'current_draft_sha256': fingerprint([asdict(r) for r in snapshot]),
                        'capacity_preflight_sha256': fingerprint(preflight)}

    diagnosis_item = {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': ['OK', 'ISSUE']}, 'reason': {'type': 'string', 'maxLength': 240},
        'context_ids': {'type': 'array', 'maxItems': 8, 'items': {'type': 'integer', 'enum': sorted(all_ids)}}},
        'required': ['status', 'reason', 'context_ids'], 'additionalProperties': False}
    for start in range(0, len(draft), batch_size):
        rows = draft[start:start + batch_size]
        body = {str(i): {'global_id': row.index, 'timestamp': row.ts_line, 'chinese': row.text}
                for i, row in enumerate(rows, 1)}
        parsed, receipt = query('diagnose', rows, body,
            RECALL_DIAGNOSE if diagnosis_policy == 'recall-precheck' else DIAGNOSE, draft,
            _record_schema(rows, diagnosis_item), critic_budget, 0,
            lambda raw: parse_diagnosis(raw, rows, all_ids))
        diagnostics.update(parsed); receipts.update({row.index: receipt for row in rows})
    issues = {key: {'issue_id': 'cue-' + str(key), 'global_id': key, **item}
              for key, item in diagnostics.items() if item['status'] == 'ISSUE'}
    active = set(issues)
    resolution_item = {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': list(STATUSES)},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240}},
        'required': ['status', 'reason'], 'additionalProperties': False}
    if diagnosis_policy == 'recall-precheck' and issues:
        precheck_item = deepcopy(resolution_item)
        precheck_item['properties']['status']['enum'] = list(PRECHECK_STATUSES)
        candidates = [row for row in draft if row.index in issues]
        for start in range(0, len(candidates), batch_size):
            rows = candidates[start:start + batch_size]
            body = {str(i): {'issue': issues[row.index], 'global_id': row.index,
                'timestamp': row.ts_line, 'original_chinese': row.text,
                'current_chinese': row.text} for i, row in enumerate(rows, 1)}
            checked, receipt = query('precheck', rows, body, PRECHECK, draft,
                _record_schema(rows, precheck_item), critic_budget, 0,
                lambda raw: parse_precheck(raw, rows))
            initial_prechecks.update(checked)
            for row in rows:
                precheck_receipts[row.index] = receipt
                histories[row.index].append({'cycle': 0, **checked[row.index],
                    'assembled_draft_sha256': fingerprint([asdict(r) for r in draft]),
                    'cue_text_sha256': fingerprint(row.text), 'repair_receipt': None,
                    'resolution_receipt': receipt})
        if set(initial_prechecks) != set(issues):
            raise CoherenceError('Incomplete unchanged-draft candidate precheck')
        previous = deepcopy(initial_prechecks)
        active = {key for key, item in previous.items() if item['status'] == 'persists'}

    for cycle in range(1, max_repair_retries + 2):
        if not active:
            break
        snapshot = deepcopy(final)
        positions = [i for i, row in enumerate(snapshot) if row.index in active]
        repair_receipts = {}
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start + batch_size]; rows = [snapshot[i] for i in selected]
            body = {str(i): {'issue': issues[row.index], 'global_id': row.index,
                'timestamp': row.ts_line, 'chinese': row.text, 'original_chinese': original[row.index].text,
                'previous_resolution': previous.get(row.index)} for i, row in enumerate(rows, 1)}
            schema = _request_config(config, rows).extra_payload['response_format']['schema']
            values, receipt = query('repair', rows, body, REPAIR, snapshot, schema, edit_budget, cycle,
                                    lambda raw: _parse_response(raw, rows))
            for position, value in zip(selected, values):
                final[position] = replace(snapshot[position], text=value)
                attempts[final[position].index] += 1
                repair_receipts[final[position].index] = receipt
        # All issue checks see the same completely assembled candidate, never an
        # intermediate batch. Previously resolved/withdrawn issues are rechecked.
        checked_rows = [row for row in final if row.index in issues]
        current = {}
        for start in range(0, len(checked_rows), batch_size):
            rows = checked_rows[start:start + batch_size]
            body = {str(i): {'issue': issues[row.index], 'global_id': row.index,
                'timestamp': row.ts_line, 'original_chinese': original[row.index].text,
                'current_chinese': row.text} for i, row in enumerate(rows, 1)}
            parsed, receipt = query('check', rows, body, RESOLVE, final,
                _record_schema(rows, resolution_item), critic_budget, cycle,
                lambda raw: parse_resolution(raw, rows))
            current.update(parsed)
            for row in rows:
                histories[row.index].append({'cycle': cycle, **parsed[row.index],
                    'assembled_draft_sha256': fingerprint([asdict(r) for r in final]),
                    'cue_text_sha256': fingerprint(row.text), 'repair_receipt': repair_receipts.get(row.index),
                    'resolution_receipt': receipt})
        if set(current) != set(issues):
            raise CoherenceError('Incomplete issue-resolution coverage')
        previous = current; active = {key for key, item in current.items() if item['status'] == 'persists'}
    validate_blocks(final)
    if len(final) != len(source) or any(a.index != b.index or a.ts_line != b.ts_line for a, b in zip(source, final)):
        raise CoherenceError('Local repair changed cue ownership or geometry')
    final_hash = fingerprint([asdict(row) for row in final])
    ledger = [{'line': row.index, 'before': old.text, 'after': row.text, 'changed': row.text != old.text,
        'issue_id': issues[row.index]['issue_id'] if row.index in issues else None,
        'diagnosis_policy': diagnosis_policy,
        'precheck_status': initial_prechecks.get(row.index, {}).get('status', 'not_run'),
        'precheck_receipt': precheck_receipts.get(row.index),
        'local_diagnosis': diagnostics[row.index], 'diagnosis_receipt': receipts[row.index],
        'resolution_status': previous[row.index]['status'] if row.index in previous else 'not_flagged',
        'unresolved': row.index in active, 'repair_attempts': attempts[row.index],
        'resolution_history': histories[row.index], 'status': 'UNVERIFIED',
        'model': base.extra_payload['model'], 'run_key': run_key,
        'settings': {'base': deepcopy(base.extra_payload), 'critic_budget': critic_budget,
                     'edit_budget': edit_budget, 'max_repair_retries': max_repair_retries,
                     'context_size': context_size, 'actual_requests_in_receipts': True},
        'final_draft_sha256': final_hash,
        'source_uncertainty_cleared': False, 'source_verified': False, 'source_exposed_to_editor': False,
        'external_feedback_scope': 'score_only' if prior_score is not None else 'none'}
        for old, row in zip(draft, final)]
    run.update(complete=True, diagnosis_policy=diagnosis_policy,
               initial_precheck_draft_sha256=fingerprint([asdict(r) for r in draft]) if initial_prechecks else None,
               initial_precheck_results=initial_prechecks, final_draft_sha256=final_hash, ledger=ledger,
               unresolved_issue_ids=[issues[key]['issue_id'] for key in sorted(active)],
               local_resolution_pass=not active, independently_evaluated=False)
    write_json(cache, saved)
    return final, ledger
