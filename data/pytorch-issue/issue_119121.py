import torch.nn as nn

import torch
from torch._dynamo.utils import maybe_enable_compiled_autograd

def fn():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1, bias=False),
        torch.nn.Linear(1, 2, bias=False),
    )
    model[0].weight = torch.nn.Parameter(torch.tensor([[-0.0053,  0.3793]]))
    model[1].weight = torch.nn.Parameter(torch.tensor([[-0.8230],[-0.7359]]))

    x = torch.tensor([[-2.1788,  0.5684], [-1.0845, -1.3986]])

    out = model(x)
    loss = out.sum()
    torch.manual_seed(0)
    loss.backward()

    return (model[0].weight.grad, model[1].weight.grad)

eager_result = fn()
with maybe_enable_compiled_autograd(True):
    compiled_result = fn()

print(eager_result)
# (tensor([[5.0872, 1.2942]]),
#  tensor([[-0.2976], [-0.2976]]))

print(compiled_result)
# (tensor([[5.0872, 1.2942]]),      <-- running inductor codegen directly should output this?
#  tensor([[-1.5589], [-1.5589]]))