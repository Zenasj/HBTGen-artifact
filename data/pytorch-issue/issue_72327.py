# torch.rand(B), torch.rand(B), torch.randn(B).sign()  # Each tensor has shape (B,)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.loss = nn.MarginRankingLoss()

    def forward(self, inputs):
        input1, input2, target = inputs
        return self.loss(input1, input2, target)

def my_model_function():
    return MyModel()

def GetInput():
    B = 100  # Example batch size
    input1 = torch.rand(B)
    input2 = torch.rand(B)
    target = torch.randn(B).sign()
    return (input1, input2, target)

