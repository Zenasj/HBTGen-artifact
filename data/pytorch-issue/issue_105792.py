# torch.rand(1, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        MAXINT = 2147483647  # Original overflow trigger value (INT_MAX)
        self.len = 200  # Reduced for feasibility (originally MAXINT + 10)
        self.params = nn.ParameterList()
        for _ in range(self.len):
            param = torch.Tensor([0.0]).cuda()
            self.params.append(nn.Parameter(param))
            if _ % 100000 == 0:
                print(f"Initialized parameter {_}")  # Debug print (only triggers at 0 for self.len=200)

    def forward(self, x):
        output = x
        for i in range(len(self.params)):
            output = output + self.params[i]
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32).cuda()

