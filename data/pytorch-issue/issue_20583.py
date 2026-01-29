# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (samplesPerBatch, nChans, 32, 32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, nChans, kernelSize, padding):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(nChans, nChans, kernel_size=kernelSize, padding=padding)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    nChans = 4
    kernelSize = 3
    padding = (1, 3)  # This is the problematic padding that triggers the assertion
    return MyModel(nChans, kernelSize, padding)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    samplesPerBatch = 6
    nChans = 4
    inputShape = [samplesPerBatch, nChans, 32, 32]
    inputData = torch.ones(*inputShape, dtype=torch.float32)
    return inputData

