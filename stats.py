from typing import NamedTuple

import tinygrad
from tinygrad import Tensor
from tinygrad import dtypes
import numpy as np


class AttnStats(NamedTuple):
    entropy: Tensor   # (bsz, n_layers, n_heads)
    varentropy: Tensor    # (bsz, n_layers, n_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=Tensor.zeros((bsz, n_layers, n_heads), dtype=dtypes.float32).contiguous(),
            varentropy=Tensor.zeros((bsz, n_layers, n_heads), dtype=dtypes.float32).contiguous(),
            n_layers=n_layers,
            n_heads=n_heads
        )
    
    @property
    def avg_entropy(self):
        return self.entropy.sum(axis=-1, keepdims=False)    # avg across heads
    
    @property
    def std_error(self):
        return Tensor.sqrt(Tensor.mean(self.varentropy)) / (self.n_layers * self.n_heads)
    
    def update(self, scores: Tensor, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = Tensor.softmax(scores, axis=-1)
        new_entropy = -Tensor.sum(Tensor.where(probs > 0, probs * Tensor.log(probs), 0), axis=-1)
        new_varentropy = Tensor.sum(probs * (Tensor.log(probs) + new_entropy.unsqueeze(-1))**2, axis=-1)

        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy
    
        return self
