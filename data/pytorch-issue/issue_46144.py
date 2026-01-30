import torch.nn as nn

py
import numpy as np
import torch


torch.manual_seed(42)


class ComplexLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.re_weights = torch.nn.Linear(1, 1)
        self.im_weights = torch.nn.Linear(1, 1)

    def forward(self, x):
        multiplied = torch.view_as_complex(torch.stack(
            [self.re_weights(x.real), self.im_weights(x.imag)], dim=-1))
        # NOTE: Change between 1 and 0 to select any of the cases.
        if 0:
            # Return magnitude
            if 1:
                # "My" definition
                return (multiplied.real ** 2 + multiplied.imag ** 2).sqrt()
            else:
                # PyTorch definition
                return multiplied.abs()
        elif 1:
            # Return angle
            if 1:  # <== change this to 0 then it diverges
                # "My" definition
                return multiplied.imag.atan2(multiplied.real)
            else:
                # PyTorch definition
                return multiplied.angle()
        else:
            # Return sum of magnitude + angle
            if 1:
                # "My" definition
                return (multiplied.real ** 2 + multiplied.imag ** 2).sqrt() + multiplied.imag.atan2(multiplied.real)
            else:
                # PyTorch definition
                return multiplied.abs() + multiplied.angle()


net = ComplexLinear()

x = torch.from_numpy(np.array([5 + 3j], dtype="complex64"))
y = torch.from_numpy(np.array([0.3], dtype="float32"))

for i in range(10000):
    res = net(x)
    loss = torch.nn.functional.mse_loss(y, res)
    net.zero_grad()
    loss.backward()
    #print(res, net.re_weights.weight.grad)
    print("\r", "step", i, res, end="")
    with torch.no_grad():
        for param in net.parameters():
            if param.requires_grad:
                param -= 1e-3 * param.grad
print()