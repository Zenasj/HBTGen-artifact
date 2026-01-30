import torch

@torch.compile
def f(x):
    s = "Hello" + x
    print(s)

f(", world!")

from torch._dynamo.eval_frame import _debug_get_cache_entry_list

name = ""
for x in list(globals().keys()):
    if x.startswith("__resume"):
        name = x # should be __resume_at_14_0

code = globals()[name]

print(_debug_get_cache_entry_list(code))