from typing import NamedTuple

import tinygrad
from tinygrad import Tensor
from tinygrad import dtypes
import numpy as np


class KVCache(NamedTuple):
    k: Tensor
    v: Tensor

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        print((layers, bsz, max_seq_len, kv_heads, head_dim))
        return cls(
            k = Tensor.zeros(layers, bsz, max_seq_len, kv_heads, head_dim, dtype=dtypes.float16).contiguous(),
            v = Tensor.zeros(layers, bsz, max_seq_len, kv_heads, head_dim, dtype=dtypes.float16).contiguous(),
        )

    def update(self, xk: Tensor, xv: Tensor, layer_idx: int, cur_pos: int, n_rep: int):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (mx.array): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (mx.array): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[mx.array, mx.array, KVCache]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        xk = xk.cast(self.k.dtype)
        xv = xv.cast(self.v.dtype)
        print(xk.dtype, xv.dtype)

        insert_len = xk.shape[1]
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = xk.expand(-1, -1, n_rep, -1)
            values = xv.expand(-1, -1, n_rep, -1)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = xk.expand(-1, -1, self.k[layer_idx], -1) 
            values = xv.expand(-1, -1, self.k[layer_idx], -1) 

        return keys, values, self
