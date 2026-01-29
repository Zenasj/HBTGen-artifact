import torch
import torch.nn as nn

# torch.rand(1, dtype=torch.float32)
class TestScript(nn.Module):
    def __init__(self):
        super(TestScript, self).__init__()
    def forward(self, x):
        return (x, )

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.test_script = torch.jit.script(TestScript())
    def forward(self, x):
        return self.test_script(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

