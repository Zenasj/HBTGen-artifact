import torch

def f(x):
    return x + 1

opt_f = torch.compile(f)

_ = opt_f(torch.randn(5, 5, 5))
from torch._dynamo.utils import _debug_get_cache_entry_list
entries = _debug_get_cache_entry_list(f.__code__) # good
entries = _debug_get_cache_entry_list(opt_f.__code__) # core dump