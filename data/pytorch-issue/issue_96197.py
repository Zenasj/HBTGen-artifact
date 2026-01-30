from dataclasses import dataclass
from typing import Callable
import torch

@dataclass
class _Metadata:
    reduce_fx: Callable = torch.mean

def log():
    meta = _Metadata()

c_log = torch.compile(log)
c_log()