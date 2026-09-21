"""CPU synthetic attention-boundary proof; no checkpoint, tokenizer or audio."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import qwen_audio_attention as repair


def tiny_encoder(backend='sdpa'):
    import torch
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    cfg = Qwen3ASRAudioEncoderConfig(d_model=8, encoder_attention_heads=2, encoder_layers=2,
        encoder_ffn_dim=16, output_dim=8, downsample_hidden_size=4, n_window=50, n_window_infer=800)
    cfg._attn_implementation = backend
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(20260916)
        encoder = Qwen3ASRAudioEncoder(cfg).float().eval()
    return encoder


def tiny_attention(backend='sdpa'):
    import torch
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioAttention
    cfg = Qwen3ASRAudioEncoderConfig(d_model=8, encoder_attention_heads=2)
    cfg._attn_implementation = backend
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(20260916); attention = Qwen3ASRAudioAttention(cfg).float().eval()
    with torch.no_grad():
        for layer in (attention.q_proj, attention.k_proj): layer.weight.zero_(); layer.bias.zero_()
        for layer in (attention.v_proj, attention.out_proj):
            layer.weight.copy_(torch.eye(8)); layer.bias.zero_()
    return attention


def run_diagnostic(backend='sdpa'):
    import torch
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    started = time.monotonic(); attention = tiny_attention(backend); encoder = tiny_encoder(backend)
    # Only a config-bearing object is needed by the actual upstream mask helper.
    hidden = torch.tensor([[1.]*8]*2+[[9.]*8]*2, dtype=torch.float32, device='cpu')
    cu = torch.tensor([0,2,4], dtype=torch.int32, device='cpu')
    mask = Qwen3ASRAudioEncoder._prepare_attention_mask(encoder, hidden, cu)
    with torch.inference_mode():
        current = attention(hidden, cu)
        masked = attention(hidden, cu, attention_mask=mask)
        # Independent segmented reference for Q=K=0: each block's arithmetic mean.
        reference = torch.cat([hidden[begin:end].mean(0,keepdim=True).expand(end-begin,-1)
                               for begin,end in zip(cu.tolist(),cu.tolist()[1:])])
        perturbed = hidden.clone(); perturbed[2:] += 8
        current_changed = attention(perturbed,cu)
        masked_changed = attention(perturbed,cu,attention_mask=mask)
        single = attention(hidden,torch.tensor([0,4],dtype=torch.int32))
        single_mask = encoder._prepare_attention_mask(hidden,torch.tensor([0,4],dtype=torch.int32))
        single_masked = attention(hidden,torch.tensor([0,4],dtype=torch.int32),attention_mask=single_mask)
        features = torch.linspace(-.01,.01,128*1200,dtype=torch.float32).reshape(128,1200)
        with repair.scoped_window_attention(encoder) as attestation:
            encoded = encoder(features,feature_lens=torch.tensor([1200],dtype=torch.int64))
    metrics = {'unmasked_output_first_column':current[:,0].tolist(),
        'masked_output_first_column':masked[:,0].tolist(), 'reference_first_column':reference[:,0].tolist(),
        'unmasked_reference_max_abs':float((current-reference).abs().max()),
        'masked_reference_max_abs':float((masked-reference).abs().max()),
        'unmasked_first_block_leakage':float((current_changed[:2]-current[:2]).abs().max()),
        'masked_first_block_leakage':float((masked_changed[:2]-masked[:2]).abs().max()),
        'single_window_max_abs':float((single-single_masked).abs().max())}
    repair._require(metrics['unmasked_reference_max_abs'] > 1 and metrics['unmasked_first_block_leakage'] > 1
        and metrics['masked_reference_max_abs'] <= 1e-6 and metrics['masked_first_block_leakage'] <= 1e-6
        and metrics['single_window_max_abs'] <= 1e-6, 'Synthetic attention hypothesis did not hold')
    repair.validate_attestation(attestation,feature_frames=1200,config=repair.encoder_config(encoder))
    repair._require(not any(m._forward_pre_hooks or m._forward_hooks
        for m in [encoder,*encoder.layers,*[x.self_attn for x in encoder.layers]]), 'Scoped hooks leaked')
    result = {'version':'qwen-audio-attention-synthetic-1','synthetic_only':True,
        'checkpoint_loads':0,'tokenizer_loads':0,'audio_inputs':0,'pretrained_model_inference_calls':0,
        'synthetic_neural_forwards':True,'device':'cpu','dtype':'float32',
        'attention_class':type(attention).__module__+'.'+type(attention).__name__,
        'attention_backend':attention.config._attn_implementation,
        'attention_hidden_input_shape':list(hidden.shape), 'head_configuration':{'heads':attention.num_heads,'head_dim':attention.head_dim},
        'qkv_tensors_captured':False,'cu_seqlens':cu.tolist(), 'mask_shape':list(mask.shape),
        'metrics':metrics,'encoder_output_shape':list(encoded.last_hidden_state.shape),
        'attestation':deepcopy(attestation),'historical_backend_attested':False,
        'acoustic_accuracy_tested':False,'seconds':time.monotonic()-started,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/qwen-audio-attention-synthetic-20260916.json')
    args = parser.parse_args(); result = {'version':'qwen-audio-attention-synthetic-comparison-1',
        'backends':{name:run_diagnostic(name) for name in ('sdpa','eager')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'metrics':{name:value['metrics']
        for name,value in result['backends'].items()}},sort_keys=True))


if __name__ == '__main__': main()
