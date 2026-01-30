py
import torch

arg = torch.randn([2,1]).expand(2,2)
print(arg.stride())  # (1, 0)
print(arg.clone(memory_format=torch.preserve_format).stride())  # (2, 1)