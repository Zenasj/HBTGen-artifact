import torch
import torch.nn as nn

SEED = 0

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


model = Model().eval()
c_model = torch.compile(model)

torch.manual_seed(SEED)
x = torch.randn(1, 4, 64, 64)  # only `in_channels = 4`, `H=64`, `W=64` can trigger inconsistency
inputs = [x]

torch.manual_seed(SEED)
output = model(*inputs)
torch.manual_seed(SEED)
c_output = c_model(*inputs)


print(torch.max(torch.abs(output - c_output)))
print(torch.allclose(output, c_output))