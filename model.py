import math
import struct
from tinygrad import TinyJit
from typing import Optional, Tuple

import tinygrad
from tinygrad import Tensor
from tinygrad import dtypes
import numpy as np
from config import ModelParams
from kvcache import KVCache
from stats import AttnStats
from weights import XfmrWeights, LayerWeights
from utils import complexarray


float32_max = struct.unpack('f', struct.pack('I', 0x7f7fffff))[0]
DEFAULT_MAX_VALUE = -0.7 * float32_max

def rms_norm(x: Tensor, w: Tensor, eps: float = 1e-10) -> Tensor:
    return x * (x.square().mean(-1, keepdim=True).cast(dtypes.float32).add(eps).rsqrt().cast(x.dtype)) * w


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: complexarray, dtype: dtypes = dtypes.float32) -> Tuple[Tensor, Tensor]:
    reshape_xq = xq.cast(dtypes.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.cast(dtypes.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = complexarray(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = complexarray(reshape_xk[..., 0], reshape_xk[..., 1])
    fc_expanded = freqs_cis.expand_dims(0).expand_dims(2)
    xq_out = xq_ * fc_expanded
    xk_out = xk_ * fc_expanded
    xq_out = xq_out.real.stack(xq_out.imag, dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_out.real.stack(xk_out.imag, dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.cast(dtype), xk_out.cast(dtype)


def attention(x: Tensor, layer_weights: LayerWeights, model_params: ModelParams,
              cur_pos: int, layer_idx: int, freqs_cis: complexarray, kvcache: KVCache,
              attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, KVCache, Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = Tensor.matmul(x, layer_weights.wq.T).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = Tensor.matmul(x, layer_weights.wk.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = Tensor.matmul(x, layer_weights.wv.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    print(xq.shape, xk.shape, xv.shape)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = xq.reshape(bsz, model_params.n_local_heads, -1, model_params.head_dim) # (bs, n_heads, seqlen, head_dim)
    keys = keys.reshape(bsz, model_params.n_local_kv_heads, model_params.head_dim, -1) # (bs, n_kv_heads, head_dim, seqlen)
    values = values.reshape(bsz, model_params.n_local_kv_heads, -1, model_params.head_dim) # (bs, n_kv_heads, seqlen, head_dim)
    print(xq.shape, keys.shape, values.shape)
    scores = Tensor.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.cast(dtypes.float32)  # always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = Tensor.where(scores != 0.0, scores, DEFAULT_MAX_VALUE)
    padded_logits = Tensor.where((mask >= DEFAULT_MAX_VALUE * 0.5), scores, DEFAULT_MAX_VALUE)
    scores = Tensor.softmax(padded_logits, axis=-1).astype(x.dtype)
    output = Tensor.matmul(scores, values)
    output = output.permute(0, 2, 1).reshape(xq.shape[0], xq.shape[2], -1)
    out = Tensor.matmul(output, layer_weights.wo.T)
    return out, kvcache, pre_scores


def feed_forward(x: Tensor, layer_weights: LayerWeights) -> Tensor:
    return Tensor.matmul(Tensor.silu(Tensor.matmul(x, layer_weights.w1.T)) * Tensor.matmul(x, layer_weights.w3.T), layer_weights.w2.T)


def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: Tensor, 
         cur_pos: int, freqs_cis: complexarray, kvcache: KVCache, 
         attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, KVCache, Tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = Tensor.matmul(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
    return logits, kvcache, scores, attn_stats
