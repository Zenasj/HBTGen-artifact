python
import torch
pt = 'libtorch_0.pt'
jit_loaded = torch.jit.load(pt)
jit_loaded.eval()
dummy_input = torch.randn(1, 1, 28, 28)
target = jit_loaded(dummy_input)