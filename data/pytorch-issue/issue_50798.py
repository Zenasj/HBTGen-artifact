# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load Shufflenet v2 model as per the test case in the issue
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return the wrapped Shufflenet model
    return MyModel()

def GetInput():
    # Generate random input matching the model's expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

