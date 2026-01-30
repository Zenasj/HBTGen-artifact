import torch.nn as nn

import torch
import torch._inductor.config
torch._inductor.config.trace.enabled = True
torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.l(x))

m = ToyModel().to(device="cuda:0")

m = torch.compile(m)
input_tensor = torch.randn(100).to(device="cuda:0")
out = m(input_tensor)