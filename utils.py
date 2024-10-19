import tinygrad 
from tinygrad import Tensor
from tinygrad import dtypes
import numpy as np

from typing import *


COLORS = {
    "lelv": "green",    # low ent, low vent
    "hehv": "red",      # high ent, high vent
    "helv": "magenta",  # high ent, low vent
    "lehv": "yellow",   # low ent, high vent
    "ada": "nocolor"    # else
}

DISPLAY_STR_COLOR = '<span style="color: {color}" data-tooltip="{data}">{decoded}</span'
DISPLAY_STR_NORMAL = '<span data-tooltip="{data}">{decoded}</span>'


def create_display_tooltip(decoded: str, color: str, metrics: dict):
    mstr = ""
    for k, v in metrics.items():
        mstr += f"{k}: {v}\n"
    if color != "nocolor":
        return DISPLAY_STR_NORMAL.format(data=mstr.strip(), decoded=decoded)
    else:
        return DISPLAY_STR_COLOR.format(color=color, data=mstr.strip(), decoded=decoded)        


class complexarray:
    """
    a smol hacky class for complex arrays using mlx functions
    because mlx does not have complex arrays apis
    """
    def __init__(self, real: Tensor, imag: Tensor):
        self.real = real
        self.imag = imag

    @property
    def shape(self):
        return self.real.shape
    
    @classmethod
    def from_polar(cls, r: Tensor, theta: Tensor) -> 'complexarray':
        return cls(r * Tensor.cos(theta), r * Tensor.sin(theta))

    def __add__(self, other: 'complexarray'):    # add two complex arrays
        return complexarray(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'complexarray'):   # multiply two complex arrays
        return complexarray(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __abs__(self):
        return Tensor.sqrt(self.real**2 + self.imag**2)
    
    def conj(self):
        return complexarray(self.real, -self.imag)
    
    def exp(self):
        """
        Exponential of a complex array.
        For a complex number `z = a + ib` it is defined as:

        `exp(z) = exp(a) * (cos(b) + i * sin(b))`
        """
        r = Tensor.exp(self.real)
        return complexarray(r * Tensor.cos(self.imag), r * Tensor.sin(self.imag))
    
    def expand_dims(self, axis: int):
        """
        Adds new axis at the given axis
        """
        return complexarray(self.real.unsqueeze(axis), self.imag.unsqueeze(axis))
    
    def __getitem__(self, key):
        return complexarray(self.real[key], self.imag[key])


if __name__ == "__main__":
    x, y = Tensor.rand(3, 4), Tensor.rand(3, 4)
    f = complexarray(x, y)
    new = f[1:2]
    print(f.real.numpy())
    print(f.imag.numpy())
    print(new.real.numpy())
    print(new.imag.numpy())
