py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_0_0 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.model_0_1 = nn.BatchNorm2d(32)
        self.model_0_2 = nn.ReLU(inplace=True)

    def forward(self, input_1):
        model_0_0 = self.model_0_0(input_1)
        model_0_1 = self.model_0_1(model_0_0)
        model_0_2 = self.model_0_2(model_0_1)
        return model_0_2

model = Model()
model.cpu()
model.train()

torch.quantization.fuse_modules(model, [['model_0_0', 'model_0_1', 'model_0_2']])