# (torch.rand(B, C, dtype=torch.float), torch.rand(B, C, dtype=torch.float))  # Input shape: tuple of two tensors
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        output, target = inputs
        probs = F.softmax(output, dim=1)
        loss = -(target * torch.log(probs)).mean()
        loss = loss.reshape(1)  # Fix empty shape for Java compatibility
        batch_size = output.size(0)
        num_classes = target.size(1)
        loss_grad = (probs - target) / (batch_size * num_classes)
        return probs, loss, loss_grad

def my_model_function():
    return MyModel()

def GetInput():
    B = 32  # Example batch size
    C = 10  # Example number of classes
    output = torch.rand(B, C, dtype=torch.float)
    target = torch.rand(B, C, dtype=torch.float)
    return (output, target)

