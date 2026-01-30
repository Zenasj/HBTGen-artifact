import torch

self = torch.randn([1,1], dtype=torch.float16)
self.sort(stable=None ,dim=0, descending=False)

3
import torch

x = torch.ones(1)
x.sort(stable=None)