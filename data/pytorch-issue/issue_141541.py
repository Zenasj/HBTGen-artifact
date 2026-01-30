import torch
import torch.nn as nn

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=32)

    def forward(self, x):
        x = self.gn(x)
        return x


model = Model().eval()
c_model = torch.compile(model)

x = torch.randn(1, 32, 128, 128, 128)

inputs = [x]

output = model(*inputs)
c_output = c_model(*inputs)

print(torch.max(torch.abs(output - c_output)))
print(torch.allclose(output, c_output, 1.3e-6, 1e-5))