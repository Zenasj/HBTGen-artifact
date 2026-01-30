import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm([10000, 1000])  # LayerNorm for 2D input

    def forward(self, x):
        x = self.ln(x)
        return x


model = Model().eval()

x = torch.randn(1, 3, 10000, 1000)  # As `H` and `W` increase, the error might be amplified

inputs = [x]

c_model = torch.compile(model)

output = model(*inputs)

c_output = c_model(*inputs)

print(torch.allclose(output, c_output, 1e-5, 1e-5))  # loose check in fp32
print(torch.max(torch.abs(output - c_output)))

fp_64_ref = c_model(x.double())
print("Eager divergence", torch.max(torch.abs(output - fp_64_ref)))
print("Compile divergence divergence", torch.max(torch.abs(c_output - fp_64_ref)))