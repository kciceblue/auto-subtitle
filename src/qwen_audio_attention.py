"""Scoped block-mask repair for the pinned installed Qwen audio encoder only.

No checkpoint is loaded here. Temporary instance hooks pass the encoder's own
mask into each layer and attest its arrival at the actual attention interface.
The context must enclose exactly one encoder call. Installed files/classes and
all other model instances remain untouched; hooks are removed on failure too.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

VERSION = 'qwen-audio-window-mask-1'
_PINS = {
    ('qwen-asr', 'qwen_asr/core/transformers_backend/modeling_qwen3_asr.py'): '2fb5d98da1933748f5117ee05ce4e7150c9ead8154fb8e25f7af3968b853adc7',
    ('qwen-asr', 'qwen_asr/core/transformers_backend/configuration_qwen3_asr.py'): 'acf6c3f1cb3dc1ea0df621a11f7df6ccca109be73b6bf51508b5052942f5bec1',
    ('transformers', 'transformers/integrations/sdpa_attention.py'): 'dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013',
    ('transformers', 'transformers/modeling_utils.py'): 'bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d',
}


def _require(value, message):
    if not value: raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_identity():
    """Read small code files/distribution metadata only; never weights or a GPU."""
    files = {}
    for (distribution, relative), expected in _PINS.items():
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative)).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected, 'Installed attention source differs from the audited implementation')
        files[str(path)] = actual
    path = Path(__file__).resolve(); files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'files': files, 'versions': {name: importlib.metadata.version(name)
            for name in ('qwen-asr', 'transformers', 'torch')}}


def encoder_config(encoder):
    cfg = encoder.config
    _require(type(encoder).__name__ == 'Qwen3ASRAudioEncoder'
        and type(encoder).__module__ == 'qwen_asr.core.transformers_backend.modeling_qwen3_asr', 'Unsupported encoder class')
    result = {'backend': cfg._attn_implementation, 'layers': len(encoder.layers),
        'hidden_size': cfg.d_model, 'heads': cfg.encoder_attention_heads,
        'head_dim': cfg.d_model//cfg.encoder_attention_heads, 'n_window': cfg.n_window,
        'n_window_infer': cfg.n_window_infer, 'max_source_positions': cfg.max_source_positions}
    _validate_config(result)
    _require(result['layers'] == cfg.encoder_layers and all(
        layer.self_attn.config._attn_implementation == result['backend']
        and layer.self_attn.num_heads == result['heads'] and layer.self_attn.head_dim == result['head_dim']
        for layer in encoder.layers), 'Encoder layers do not share the declared backend/shape')
    return result


def _validate_config(cfg):
    _require(type(cfg) is dict and set(cfg) == {'backend', 'layers', 'hidden_size', 'heads', 'head_dim',
        'n_window', 'n_window_infer', 'max_source_positions'}, 'Unexpected encoder configuration fields')
    _require(cfg['backend'] in ('sdpa', 'eager') and all(type(v) is int and v > 0
        for k,v in cfg.items() if k != 'backend') and cfg['heads']*cfg['head_dim'] == cfg['hidden_size']
        and cfg['n_window'] == 50 and cfg['n_window_infer'] == 800,
        'Only the audited non-FA2 encoder window policy is supported')


def expected_geometry(feature_frames, cfg):
    _validate_config(cfg)
    _require(type(feature_frames) is int and feature_frames > 0, 'Invalid feature-frame count')
    def length(n):
        rest = n % 100
        return ((((rest-1)//2+1)-1)//2+1-1)//2+1+(n//100)*13
    total = length(feature_frames)
    width = length(min(feature_frames, 100))*(cfg['n_window_infer']//100)
    boundaries = list(range(0, total, width))+[total]
    return total, boundaries


def _mask_hash(length, boundaries, dtype):
    # Independent CPU byte reference, not a cached copy of the injected tensor.
    formats = {'float16': ('e', -65504.), 'float32': ('f', -3.4028234663852886e38)}
    _require(dtype in formats, 'Unsupported mask dtype')
    fmt, minimum = formats[dtype]; blocked = struct.pack('<'+fmt, minimum); zero = struct.pack('<'+fmt, 0.)
    size = len(blocked); raw = bytearray(blocked*(length*length))
    for begin,end in zip(boundaries,boundaries[1:]):
        for row in range(begin,end): raw[(row*length+begin)*size:(row*length+end)*size] = zero*(end-begin)
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def scoped_window_attention(encoder):
    """Yield a sealed JSON attestation for exactly one repaired encoder call."""
    import torch
    cfg = encoder_config(encoder); identity = source_identity(); handles = []; active = None
    modules = [encoder, *encoder.layers, *[layer.self_attn for layer in encoder.layers]]
    _require(all(not module._forward_pre_hooks and not module._forward_hooks for module in modules),
             'Existing/nested encoder hooks would obscure the scoped repair')
    audit = {'version': VERSION, 'source_identity': identity, 'encoder_config': cfg,
             'encoder_calls': 0, 'calls': [], 'status': 'running'}

    def begin(module, args, kwargs):
        nonlocal active
        _require(active is None and audit['encoder_calls'] == 0, 'Exactly one non-reentrant encoder call is allowed')
        lengths = kwargs.get('feature_lens', args[1] if len(args)>1 else None)
        _require(isinstance(lengths, torch.Tensor) and lengths.numel() == 1, 'One audio feature length is required')
        frames = int(lengths.item()); total, boundaries = expected_geometry(frames, cfg)
        record = {'feature_frames': frames, 'sequence_length': total, 'hidden_size': cfg['hidden_size'],
                  'cu_seqlens': boundaries, 'mask_shape': [1,1,total,total], 'mask_dtype': None,
                  'mask_sha256': None, 'device': None, 'layer_calls': [], 'complete': False}
        audit['encoder_calls'] += 1; audit['calls'].append(record)
        active = {'record': record, 'mask': None, 'next_layer': 0, 'next_attention': 0}

    def layer_hook(index):
        def inject(module, args, kwargs):
            _require(active is not None and active['next_layer'] == index, 'Encoder layer order changed')
            _require(len(args) == 2 and kwargs.get('attention_mask') is None, 'Unexpected existing layer mask/arguments')
            hidden, boundaries = args; record = active['record']
            _require(list(hidden.shape) == [record['sequence_length'],cfg['hidden_size']]
                and boundaries.dtype == torch.int32 and boundaries.tolist() == record['cu_seqlens'], 'Encoder window geometry changed')
            if active['mask'] is None:
                mask = encoder._prepare_attention_mask(hidden, boundaries)
                dtype = str(mask.dtype).removeprefix('torch.')
                digest = hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                _require(list(mask.shape) == record['mask_shape']
                    and digest == _mask_hash(record['sequence_length'], record['cu_seqlens'], dtype), 'Existing helper did not produce the exact block mask')
                record.update(mask_dtype=dtype, mask_sha256=digest, device=str(hidden.device)); active['mask'] = mask
            _require(hidden.device == active['mask'].device and hidden.dtype == active['mask'].dtype,
                     'Encoder layer dtype/device changed')
            active['next_layer'] += 1
            return args, {**kwargs, 'attention_mask': active['mask']}
        return inject

    def attention_hook(index):
        def observe(module, args, kwargs):
            _require(active is not None and active['next_attention'] == index
                and active['next_layer'] == index+1 and not args
                and kwargs.get('attention_mask') is active['mask'], 'Block mask did not reach the actual attention call')
            _require(module.config._attn_implementation == cfg['backend']
                and list(kwargs['hidden_states'].shape) == [active['record']['sequence_length'],cfg['hidden_size']]
                and kwargs['cu_seqlens'].tolist() == active['record']['cu_seqlens'], 'Attention dispatch or shape changed')
            active['record']['layer_calls'].append({'layer_index': index, 'backend': cfg['backend'],
                'heads': module.num_heads, 'head_dim': module.head_dim})
            active['next_attention'] += 1
        return observe

    def finish(module, args, kwargs, output):
        nonlocal active
        if active is None: return
        if output is not None:
            _require(active['next_layer'] == active['next_attention'] == cfg['layers'], 'Incomplete layer/mask coverage')
            active['record']['complete'] = True
        active = None

    success = False
    try:
        handles.append(encoder.register_forward_pre_hook(begin, with_kwargs=True))
        handles.append(encoder.register_forward_hook(finish, with_kwargs=True, always_call=True))
        for index,layer in enumerate(encoder.layers):
            handles.append(layer.register_forward_pre_hook(layer_hook(index), with_kwargs=True))
            handles.append(layer.self_attn.register_forward_pre_hook(attention_hook(index), with_kwargs=True))
        yield audit
        _require(audit['encoder_calls'] == 1 and audit['calls'][0]['complete'], 'A complete single encoder call is required')
        success = True
    finally:
        for handle in reversed(handles): handle.remove()
        audit['status'] = 'complete' if success else 'failed'
        audit['attestation_sha256'] = _hash(audit)


def validate_attestation(value, *, feature_frames, config, source=None):
    """Independently check numeric mask/coverage; caller binds the native receipt."""
    _validate_config(config)
    _require(type(value) is dict and set(value) == {'version','source_identity','encoder_config','encoder_calls',
        'calls','status','attestation_sha256'}, 'Unexpected attention attestation fields')
    _require(value['version'] == VERSION and value['status'] == 'complete'
        and type(value['encoder_calls']) is int and value['encoder_calls'] == 1
        and type(value['calls']) is list and len(value['calls']) == 1
        and value['attestation_sha256'] == _hash({k:v for k,v in value.items() if k != 'attestation_sha256'})
        and _hash(value['source_identity']) == _hash(source_identity() if source is None else source)
        and _hash(value['encoder_config']) == _hash(config), 'Attention attestation/source/config changed')
    total, boundaries = expected_geometry(feature_frames, config); record = value['calls'][0]
    _require(type(record) is dict and set(record) == {'feature_frames','sequence_length','hidden_size','cu_seqlens',
        'mask_shape','mask_dtype','mask_sha256','device','layer_calls','complete'}, 'Unexpected encoder-call fields')
    expected = {'feature_frames':feature_frames,'sequence_length':total,'hidden_size':config['hidden_size'],
        'cu_seqlens':boundaries,'mask_shape':[1,1,total,total], 'complete':True,
        'layer_calls':[{'layer_index':i,'backend':config['backend'],'heads':config['heads'],'head_dim':config['head_dim']}
                       for i in range(config['layers'])]}
    _require(all(_hash(record[k]) == _hash(v) for k,v in expected.items())
        and record['mask_sha256'] == _mask_hash(total,boundaries,record['mask_dtype'])
        and type(record['device']) is str and bool(record['device']), 'Mask bytes, geometry or full layer coverage changed')
    return deepcopy(record)
