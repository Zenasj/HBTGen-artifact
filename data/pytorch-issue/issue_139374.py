import torch.nn as nn

import torch
from triton.testing import do_bench
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, noop_mask, BlockMask
import torch.nn.functional as F
import functools

torch.manual_seed(0)

import torch
torch.set_default_device('cuda')

def sliding_window(b, h, q_idx, kv_idx, val):
    return (q_idx - kv_idx).abs() < val

sliding_window2 = functools.partial(sliding_window, val=torch.randn(()))
torch.compile(create_block_mask, fullgraph=True)(sliding_window2, None, None, 1024, 1024)

py
def _get_mod_type(fn: Callable) -> _ModificationType:
    num_defaults = 0 if fn.__defaults__ is None else len(fn.__defaults__)
    num_positional_args = fn.__code__.co_argcount - num_defaults
    ...