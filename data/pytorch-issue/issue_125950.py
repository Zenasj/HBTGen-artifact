# a.py
import torch
from math import sqrt

# This file is used by test_class_type_closure

class MySqrtClass:
    value: float

    def __init__(self):
        self.value = 4.0

    def get_sqrt_inverse(self) -> float:
        return sqrt(1.0 / self.value)

# b.py
import torch
from a import MySqrtClass

torch.jit.script(MySqrtClass)