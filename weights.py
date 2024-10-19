from typing import List, NamedTuple
import os
import tinygrad
from tinygrad import Tensor, nn
from tinygrad import dtypes
import numpy as np

from pathlib import Path


class LayerWeights(NamedTuple):
    wq: Tensor
    wk: Tensor
    wv: Tensor
    wo: Tensor
    w1: Tensor
    w2: Tensor
    w3: Tensor
    ffn_norm: Tensor
    attention_norm: Tensor


class XfmrWeights(NamedTuple):
    tok_embeddings: Tensor
    norm: Tensor
    output: Tensor
    layer_weights: List[LayerWeights]


def load_weights(ckpt_dir: Path, n_layers: int = 16):
    """
    MLX will use metal gpu by default
    """
    w = {}
    layer_weights = []
    for file in ckpt_dir.glob("*.npy"):
        name = os.path.splitext(os.path.basename(file))[0]
        weight = Tensor(np.load(file)).cast(dtypes.float16)
        w[name] = weight
    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            wq=w[f'layers.{i}.attention.wq.weight'],
            wk=w[f'layers.{i}.attention.wk.weight'],
            wv=w[f'layers.{i}.attention.wv.weight'],
            wo=w[f'layers.{i}.attention.wo.weight'],
            w1=w[f'layers.{i}.feed_forward.w1.weight'],
            w2=w[f'layers.{i}.feed_forward.w2.weight'],
            w3=w[f'layers.{i}.feed_forward.w3.weight'],
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
            attention_norm=w[f'layers.{i}.attention_norm.weight'],
        ))
    xfmr_weights = XfmrWeights(
        tok_embeddings=w['tok_embeddings.weight'],
        norm=w['norm.weight'],
        output=w['output.weight'],
        layer_weights=layer_weights
    )

    return xfmr_weights
