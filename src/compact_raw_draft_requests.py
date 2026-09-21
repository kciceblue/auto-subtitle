"""Pure compact representation of A1, with source-bound B -> A -> R replay.

Only the model-visible body representation and a fixed decoding prefix change.
The sidecar contains the codec version and A's removed-field/clauses audit, not
backup retained evidence. No file access, inference or registration occurs here.
"""
from __future__ import annotations

from copy import deepcopy

from src import compact_raw_evidence as codec
from src import evidence_context as ec
from src import raw_evidence_draft_requests as raw
from src.workflow_state import fingerprint

VERSION = 'compact-raw-draft-requests-1'
OWNER_IDS = raw.OWNER_IDS
MAX_CHINESE_CHARS = raw.MAX_CHINESE_CHARS
writer_schema = raw.writer_schema
build_verifier_request = raw.build_verifier_request
build_global_request = raw.build_global_request
VERIFIER_INSTRUCTION = raw.VERIFIER_INSTRUCTION
GLOBAL_INSTRUCTION = raw.GLOBAL_INSTRUCTION

DECODING_INSTRUCTION = (
    '本请求正文使用紧凑表格表示。下文任务指令中的字段路径均指解码后的逻辑结构，'
    '例如source_evidence.views，而不是紧凑正文的顶层路径。按以下固定规则读取：'
    'source_rows、observers、view_types、views及各view内readings的数组行，分别按columns中同名列定义逐项对应。'
    'view_type_ref是view_types的从0开始的行索引，展开为kind和transform；'
    'sample_rate_ref是sample_rates的从0开始的索引，展开为sample_rate；'
    'attributes_ref是reading_attributes的从0开始的索引，将该对象中实际存在的字段原样并入该reading。'
    'reading的observer_instance是observers表中id列的标识符，不是行索引；按id查得family，'
    '保留observer_instance，并从所在view继承owner_id。'
    '每条reading的observation_id和text直接写在该行，text是完整原文，不是引用；相同文字的不同reading仍是不同记录。'
    'view的时间区间字符串、null、duplicate_geometry和数组顺序原样保留；可选字段缺失与null或false不同。'
    '解码后把source_rows、original_context、views、authority、temporal_authority、scope_rules和additional_literals'
    '放入source_evidence，focus_owner_ids仍在逻辑正文顶层。'
    'version、reference_index_base、columns和各定义表只说明上述编码，不增加观察或独立证据。\n\n'
)
WRITER_INSTRUCTION = DECODING_INSTRUCTION + raw.WRITER_INSTRUCTION


def _raw_request(request: dict) -> dict:
    ec._keys(request, {'instruction', 'body', 'schema'})
    ec._require(request['instruction'] == WRITER_INSTRUCTION,
                'Compact raw writer instruction changed')
    decoded = {'instruction': request['instruction'][len(DECODING_INSTRUCTION):],
               'body': codec.decode_body(request['body']), 'schema': deepcopy(request['schema'])}
    raw.validate_writer_request(decoded)
    return decoded


def _raw_audit(audit: dict) -> dict:
    ec._keys(audit, {'codec_version', 'raw_audit'})
    ec._require(audit['codec_version'] == codec.VERSION, 'Compact raw audit codec changed')
    return audit['raw_audit']


def validate_writer_request(request: dict) -> dict[str, int]:
    """Validate closed B input; source authenticity requires validate_ablation."""
    return raw.validate_writer_request(_raw_request(request))


def reconstruct_request(request: dict, audit: dict) -> dict:
    """Decode actual visible B fields, then restore R using A's removed-only audit."""
    return raw.reconstruct_request(_raw_request(request), _raw_audit(audit))


def validate_ablation(request: dict, audit: dict, pack: dict, bound: dict, identity_map: dict,
                      *, projection: dict | None = None) -> dict[str, int]:
    """Authenticate actual decoded evidence and removed values against pinned R inputs."""
    return raw.validate_ablation(_raw_request(request), _raw_audit(audit), pack, bound,
                                identity_map, projection=projection)


def build_ablation(pack: dict, bound: dict, identity_map: dict,
                   *, projection: dict | None = None) -> dict:
    original = raw.build_ablation(pack, bound, identity_map, projection=projection)
    request = {'instruction': WRITER_INSTRUCTION, 'body': codec.encode_body(original['request']['body']),
               'schema': deepcopy(original['request']['schema'])}
    audit = {'codec_version': codec.VERSION, 'raw_audit': deepcopy(original['audit'])}
    ec._require(fingerprint(reconstruct_request(request, audit)) ==
                fingerprint(raw.reconstruct_request(original['request'], original['audit'])),
                'Compact raw request round trip changed')
    return {'version': VERSION, 'request': request, 'audit': audit}


def build_writer_request(pack: dict, bound: dict, identity_map: dict,
                         *, projection: dict | None = None) -> dict:
    """Return the compact model request; build_ablation also returns its sidecar."""
    return build_ablation(pack, bound, identity_map, projection=projection)['request']
