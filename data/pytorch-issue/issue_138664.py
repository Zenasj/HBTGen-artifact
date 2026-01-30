import torch.nn as nn

import torch 
torch._dynamo.config.capture_scalar_outputs = True

class M(torch.nn.Module):
    def forward(self, a, b1, b2, c):
        def true_fn(x):
            return x * b1.item()

        def false_fn(x):
            return x * b2.item()

        r = torch.cond(a, true_fn, false_fn, (c,))
        return r * 2


args = (
    torch.tensor(True),
    torch.tensor([4]),
    torch.tensor([4]),
    torch.randn(10, requires_grad=True),
)
# torch.export.export(M(), args, strict=False)

model = M()

expected_output = model(*args)