import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.FractionalMaxPool3d(kernel_size=(1, 1, 1), output_ratio=(0.5, 0.5, 0.5))

    def forward(self, x):
        x = self.pool(x)
        return x


model = Model().eval()
c_model = torch.compile(model)  # backend="cudagraph" is OK

x = torch.randn(1, 1, 10, 10, 10)
inputs = [x]

torch.manual_seed(0)
output1 = model(*inputs)

torch.manual_seed(0)
output2 = model(*inputs)

torch.manual_seed(0)
c_output = c_model(*inputs)

print(torch.allclose(output1, output2))
print(torch.allclose(output1, c_output))
print(torch.max(torch.abs(output1 - c_output)))