import torch
import torch._dynamo as dynamo

@dynamo.optimize("eager")
def forward(x):
    return torch.ops.aten.bincount(x)

forward(torch.randint(0, 10, (1000,)))