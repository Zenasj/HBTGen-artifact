import torch
import torch._dynamo as dynamo

@dynamo.optimize("eager")
def forward(x):
    return torch.ops.aten.lift_fresh_copy(x)

forward(torch.randn(2, 3))