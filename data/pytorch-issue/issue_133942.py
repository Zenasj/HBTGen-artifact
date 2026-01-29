# torch.rand(1024, 153, dtype=torch.float32)  # inferred input shape for x_7 (part of the input tuple)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        L_L_labels_, x_7, labels = inputs
        out = self.dropout(x_7)
        loss = self.loss_fn(out, L_L_labels_)
        predictions = torch.argmax(out, 1)
        eq = (labels == predictions)
        sum_ = eq.sum(dtype=torch.float64)
        truediv = sum_ / 1024  # batch size assumed 1024
        num_correct = truediv * 1024
        return loss, num_correct

def my_model_function():
    return MyModel()

def GetInput():
    L_L_labels_ = torch.randint(0, 10, (1024,), dtype=torch.int64)
    x_7 = torch.randn(1024, 153, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 10, (1024,), dtype=torch.int64)
    return (L_L_labels_, x_7, labels)  # returns a tuple of three tensors

