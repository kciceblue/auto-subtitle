"""Reproducible local editing recipes; external findings never enter writer inputs."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import time

from src.coherence import _context, _json
from src.config import TranslateConfig
from src.local_backend import temporary_local_writer, _weight_bundle
from src.translate import SrtBlock
from src.workflow_state import file_hash, fingerprint, write_json

VERSION = 'local-coherence-recipe-4'
MODES = {'edit-evidence', 'critic-edit', 'scan-edit', 'target-edit', 'target-critic-edit', 'scene-critic-edit', 'resolution-edit'}
SAMPLERS = {'temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'min_p', 'presence_penalty', 'frequency_penalty'}


def load_recipe(path: Path) -> dict:
    """Validate explicit configuration. Relative paths are relative to the recipe."""
    path = Path(path).resolve(strict=True)
    data = _json(path.read_text(encoding='utf-8'))
    allowed = {'version', 'stages', 'model', 'weights', 'sha256', 'server_binary',
               'context_size', 'sampler', 'critic_budget', 'batch_size', 'max_tokens',
               'admin_url', 'restore_model', 'full_swa', 'edit_budget', 'prior_score', 'shard_sha256', 'cpu_moe_layers', 'gpu_layers', 'threads', 'separate_instruction', 'resolution_diagnosis_policy'}
    if not isinstance(data, dict) or set(data) - allowed or data.get('version') != 1:
        raise ValueError('Invalid local coherence recipe version or fields')
    explicit_gpu_layers = 'gpu_layers' in data
    stages = data.get('stages')
    if not isinstance(stages, list) or not 1 <= len(stages) <= 8 or any(not isinstance(s, str) or s not in MODES for s in stages):
        raise ValueError('Recipe requires one to eight supported local refinement stages')
    if 'resolution_diagnosis_policy' in data and 'resolution-edit' not in stages:
        raise ValueError('resolution_diagnosis_policy requires a resolution-edit stage')
    if 'resolution-edit' in stages:
        policy = data.setdefault('resolution_diagnosis_policy', 'conservative')
        if not isinstance(policy, str) or policy not in {'conservative', 'recall-precheck'}:
            raise ValueError('Invalid resolution diagnosis policy')
    model = data.get('model')
    if (not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', model)
            or model.casefold() in {'gpt-6-astra', 'fable-5.1'}):
        raise ValueError('Recipe must identify its local writer model')
    for field, default, lower, upper in [('context_size', 16384, 4096, 131072),
            ('critic_budget', 2048, 0, 8192), ('edit_budget', 0, 0, 8192),
            ('cpu_moe_layers', 0, 0, 512), ('gpu_layers', 99, 0, 512), ('threads', 4, 1, 512), ('batch_size', 20, 1, 80), ('max_tokens', 4096, 512, 16384)]:
        value = data.setdefault(field, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'Invalid recipe {field}')
    if 'resolution-edit' in stages and (data['batch_size'] > 40 or data['max_tokens'] > 8192
            or max(data['critic_budget'], data['edit_budget']) > 4096):
        raise ValueError('Resolution stage requires batch_size<=40, max_tokens<=8192 and reasoning budgets<=4096')
    if max(data['critic_budget'], data['edit_budget']) >= data['max_tokens']:
        raise ValueError('Critic reasoning must leave room for its answer')
    score = data.setdefault('prior_score', None)
    if score is not None and (type(score) is not int or not -10 <= score <= 4):
        raise ValueError('Recipe prior_score must be an aggregate integer from -10 to 4')
    sampler = data.setdefault('sampler', {'temperature': 0.3})
    if not isinstance(sampler, dict) or set(sampler) - SAMPLERS:
        raise ValueError('Recipe sampler may contain only numeric sampling controls')
    for key, value in sampler.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid sampler {key}')
        if key in {'top_k', 'seed'} and type(value) is not int:
            raise ValueError(f'Sampler {key} must be an integer')
        if key in {'temperature', 'top_k', 'repeat_penalty'} and value < 0:
            raise ValueError(f'Invalid sampler {key}')
        if key in {'top_p', 'min_p'} and not 0 <= value <= 1:
            raise ValueError(f'Invalid sampler {key}')
        if key in {'presence_penalty', 'frequency_penalty'} and not -2 <= value <= 2:
            raise ValueError(f'Invalid sampler {key}')
    if type(data.setdefault('separate_instruction', False)) is not bool:
        raise ValueError('Recipe separate_instruction must be a boolean')
    if type(data.setdefault('full_swa', False)) is not bool:
        raise ValueError('Recipe full_swa must be a boolean')
    if 'weights' in data:
        if not {'server_binary', 'sha256'} <= set(data):
            raise ValueError('Temporary writer requires weights, server_binary and sha256')
        for key in ('weights', 'server_binary'):
            if not isinstance(data[key], str) or not data[key]:
                raise ValueError(f'Invalid recipe {key}')
            p = Path(data[key]).expanduser()
            p = (p if p.is_absolute() else path.parent / p).resolve(strict=True)
            if not p.is_file() or (key == 'server_binary' and not os.access(p, os.X_OK)):
                raise ValueError(f'Invalid recipe {key} file')
            data[key] = str(p)
        if not isinstance(data['sha256'], str) or not re.fullmatch('[a-fA-F0-9]{64}', data['sha256']):
            raise ValueError('Recipe requires a SHA256 weight pin')
        data['sha256'] = data['sha256'].lower()
        _weight_bundle(Path(data['weights']), data['sha256'], data.get('shard_sha256'))
    elif explicit_gpu_layers or data['full_swa'] or data['cpu_moe_layers'] or data['threads'] != 4 or any(k in data for k in ('server_binary', 'sha256', 'shard_sha256', 'admin_url', 'restore_model')):
        raise ValueError('Temporary backend controls require weights')
    # Loopback/identity checks precede any writer call, including the existing endpoint.
    from urllib.parse import urlsplit
    admin = urlsplit(data.setdefault('admin_url', 'http://127.0.0.1:8089/admin'))
    if (admin.scheme != 'http' or admin.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or admin.username or admin.password or admin.query or admin.fragment):
        raise ValueError('Recipe admin_url must be local loopback HTTP')
    restore = data.setdefault('restore_model', 'qwen3.8-27b-dflash')
    if not isinstance(restore, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', restore):
        raise ValueError('Invalid recipe restore_model')
    return data


def recipe_binding(path: Path, config: TranslateConfig, metadata: dict) -> dict:
    recipe = load_recipe(path)
    _, original_context = _context(config)
    return {'version': VERSION, 'recipe': recipe, 'recipe_sha256': file_hash(Path(path)),
            'server_sha256': file_hash(Path(recipe['server_binary'])) if 'weights' in recipe else None,
            'original_context': fingerprint(original_context), 'metadata': fingerprint(metadata),
            'code': {name: file_hash(Path(__file__).with_name(name)) for name in
                     ('coherence_workflow.py', 'coherence.py', 'coherence_refine.py', 'coherence_resolution.py', 'local_backend.py', 'translate.py', 'config.py')}}


def refine_with_recipe(source: list[SrtBlock], draft: list[SrtBlock], metadata: dict,
                       config: TranslateConfig, cache: Path, *, expected_binding: dict | None = None
                       ) -> tuple[list[SrtBlock], list[dict]]:
    from src.coherence_refine import refine_coherence
    if config.coherence_recipe is None:
        raise ValueError('Missing local coherence recipe')
    binding = recipe_binding(config.coherence_recipe, config, metadata)
    if expected_binding is not None and binding != expected_binding:
        raise ValueError('Local coherence inputs changed after the stage fingerprint')
    recipe = binding['recipe']
    if 'resolution-edit' in recipe['stages'] and not 0 < config.timeout <= 600:
        raise ValueError('Resolution stage requires a request timeout at most600 seconds')
    key = fingerprint(binding)
    payload = {**recipe['sampler'], 'model': recipe['model'], 'max_tokens': recipe['max_tokens'],
               'cache_prompt': True}
    cfg = replace(config, extra_payload=payload, max_tokens=recipe['max_tokens'], retries=0,
                  separate_instruction=recipe.get('separate_instruction', False))
    if 'weights' in recipe:
        backend = temporary_local_writer(recipe['weights'], recipe['server_binary'],
            alias=recipe['model'], expected_sha256=recipe['sha256'], expected_shards=recipe.get('shard_sha256'),
            context_size=recipe['context_size'], full_swa=recipe['full_swa'],
            cpu_moe_layers=recipe['cpu_moe_layers'], gpu_layers=recipe.get('gpu_layers', 99), threads=recipe['threads'],
            admin_url=recipe['admin_url'], restore_model=recipe['restore_model'],
            log_path=cache.with_name(cache.stem + '.server.log'))
    else:
        backend = nullcontext((config.endpoint, {'model_alias': recipe['model'], 'managed_externally': True}))
    final = deepcopy(draft)
    ledger = [{'line': b.index, 'before': b.text, 'after': b.text, 'changed': False,
               'status': 'UNVERIFIED', 'source_uncertainty_cleared': False, 'steps': []} for b in draft]
    events = []
    start = time.monotonic()
    with backend as (endpoint, identity):
        cfg = replace(cfg, endpoint=endpoint)
        for index, mode in enumerate(recipe['stages'], 1):
            before = time.monotonic()
            request_cache = cache.with_name(f'{cache.stem}.{key[:16]}.{index}-{mode}.json')
            if mode == 'resolution-edit':
                from src.coherence_resolution import refine_resolution
                final, edits = refine_resolution(source, final, cfg, request_cache,
                    batch_size=recipe['batch_size'], critic_budget=recipe['critic_budget'],
                    edit_budget=recipe['edit_budget'], prior_score=recipe['prior_score'],
                    context_size=recipe['context_size'], max_repair_retries=2,
                    diagnosis_policy=recipe['resolution_diagnosis_policy'])
            else:
                final, edits = refine_coherence(source, final, metadata, cfg, request_cache,
                    mode=mode, batch_size=recipe['batch_size'], critic_budget=recipe['critic_budget'],
                    edit_budget=recipe['edit_budget'], prior_score=recipe['prior_score'])
            if len(edits) != len(ledger):
                raise ValueError('Local refinement returned an incomplete ledger')
            for row, edit, block in zip(ledger, edits, final):
                row['steps'].append(edit)
                row.update(after=block.text, changed=block.text != row['before'])
            events.append({'stage': mode, 'seconds': time.monotonic() - before})
    write_json(cache, {'binding': binding, 'backend': identity, 'events': events,
                       'wall_seconds': time.monotonic() - start,
                       'independently_evaluated': False, 'ledger': ledger})
    return final, ledger
