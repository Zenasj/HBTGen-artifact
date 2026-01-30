import torch.nn as nn

import torch

class Cache(torch.nn.Module):
    def forward(self, key, cache_next):
        cache_next[1,0,1] = cache_next[1,0,1]+2*key
        return key

class Cache1(torch.nn.Module):
    def forward(self, key, cache_next):
        cache_line = cache_next.select(0,1)
        cache_line[0,1] = cache_line[0,1]+2*key
        return key

class Cache3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = Cache1()
    def forward(self, key):
        cache_next = torch.zeros((2,2,2))
        self.cache(key, cache_next)
        key = key + cache_next[1,0,1]
        return key

key = torch.randint(1,10,(1,))

cache3 = Cache3().eval()
tt = torch.jit.trace(cache3, (key))
tt = torch.jit.optimize_for_inference(torch.jit.freeze(tt))
print(tt.code)

torch.onnx.export(cache3, (key),  'c3.onnx', opset_version=14, verbose=True)