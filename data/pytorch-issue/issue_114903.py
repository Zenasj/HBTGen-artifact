import torch.nn as nn

import torch

import torch._dynamo as dynamo

torch._dynamo.config.repro_after="aot"

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.var(x)

model = Model()

def test_backend_error():
    x = torch.rand((10, 3, 352, 352), dtype=torch.float16, device='cuda')
    y = model(x)
    assert not torch.isnan(y)

compiled_test_backend_error = torch.compile(test_backend_error, backend="inductor")
compiled_test_backend_error()