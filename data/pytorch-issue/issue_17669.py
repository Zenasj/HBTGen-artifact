# torch.rand(B, 50, dtype=torch.float32)  # Input shape is batch_size x 50 features
import torch
import torch.nn as nn
from torch.nn import functional as F

class MyModel(torch.jit.ScriptModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    @torch.jit.script_method
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

def my_model_function():
    # Returns the ScriptModule model with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(50, 50, dtype=torch.float32)  # Batch size 50, 50 features

