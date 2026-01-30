import torch
import torch._dynamo as dynamo


@dynamo.optimize("eager")
def forward(x):
    return int(torch.ops.aten.ScalarImplicit(x))


forward(torch.randint(low=-100, high=100, size=()))