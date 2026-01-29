# torch.rand(16,5), torch.randint(5, (16,), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, weights):
        super(MyModel, self).__init__()
        self.weights = torch.FloatTensor(weights)
        self.cross_ent_mean = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')
        self.cross_ent = nn.CrossEntropyLoss(weight=self.weights, reduction='none')

    def forward(self, inputs):
        logits, targets = inputs
        loss_a = self.cross_ent_mean(logits, targets)
        loss_b = self.cross_ent(logits, targets).mean()
        return loss_a, loss_b

def my_model_function():
    weights = [1, 2, 3, 4, 5]
    return MyModel(weights)

def GetInput():
    B, C = 16, 5
    logits = torch.randn(B, C)
    targets = torch.randint(C, (B,), dtype=torch.long)
    return (logits, targets)

