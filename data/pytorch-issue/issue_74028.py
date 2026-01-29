# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, 3, height, width)

import torch
import torch.nn as nn

class MyModel(nn.Module):
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

def my_model_function():
    model = MyModel()
    model.cpu()
    model.eval()  # Ensure the model is in eval mode for fusion
    torch.ao.quantization.fuse_modules(model, [['model_0_0', 'model_0_1', 'model_0_2']])
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 1  # Example batch size
    height = 224  # Example height
    width = 224  # Example width
    return torch.rand(batch_size, 3, height, width, dtype=torch.float32)

