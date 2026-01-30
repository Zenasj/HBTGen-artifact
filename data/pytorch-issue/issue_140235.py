import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.conv.dilation = 1  # setting dilation as an integer should now be handled

    def forward(self, x):
        return self.conv(x)

model = SimpleModel()
input_tensor = torch.randn(1, 1, 5, 5)
output = model(input_tensor)
print(model)