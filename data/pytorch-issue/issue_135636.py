import torch.nn as nn

import torch

class NewConv2d(torch.nn.Conv2d):
    def forward(self, x):
        # This would be more complicated than this
        return super().forward(x)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = NewConv2d(3, 16, 1, stride=2)

    def forward(self, x):
        return self.conv(x)


inp = torch.randn(1, 3, 224, 224, device='cuda')
model = Model()

model.eval()
model.cuda()
compiled_model = torch.compile(model, fullgraph=True, dynamic=True)
with torch.no_grad():
    compiled_model(torch.randn(1, 3, 224, 224, device='cuda'))
    compiled_model(torch.randn(1, 3, 224, 224, device='cuda'))