import os
import torch

def func(a):
 b = torch.frac(a)
 return b

os.environ["PT_HPU_LAZY_MODE"] = "0"
import habana_frameworks.torch.hpu
x = torch.randn([2, 3]).to('hpu')
compiled_func = torch.compile(func, backend="aot_hpu_training_backend")

result = compiled_func(x)
print(result)

tensor([[0., 0., 0.],
        [0., 0., 0.]], device='hpu:0')