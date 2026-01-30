import numpy as np

import torch
import triton

torch.cuda._check_bf16_tensor_supported = lambda x: False
print(torch.__version__, triton.__version__, torch.cuda.get_device_properties(0), torch.cuda.is_bf16_supported())


@torch.compile
def fn(inp, src, index):
            return inp.scatter_add(0, index, src)

with torch.device("cuda"):
  dtype = torch.bfloat16
  inp = torch.zeros(3, 5, dtype=dtype)
  src = torch.ones((2, 5), dtype=dtype)
  index = torch.tensor([[0, 1, 2, 0, 0]])

print(fn(inp, src, index))