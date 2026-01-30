import torch
import torch._dynamo as dynamo


@dynamo.optimize("eager")
def forward(x):
    return torch.ops.aten._to_copy(x)


forward(torch.randn(3, 2, 4))