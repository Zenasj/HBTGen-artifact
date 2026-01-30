import torch
import torch.nn as nn

class MyComplexCrossEntropyLoss(nn.Module):
    
    def __init__(self):
        super(MyComplexCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):       
        real_loss = nn.CrossEntropyLoss(inputs.real, targets)
        if torch.is_complex(inputs):
            imag_loss = nn.CrossEntropyLoss(inputs.imag, targets)
            return (real_loss + imag_loss)/2
        else:
            return real_loss

class MyComplexCrossEntropyLoss(nn.Module):
    
    def __init__(self):
        super(MyComplexCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):       
        
        if torch.is_complex(inputs):
            real_loss = nn.CrossEntropyLoss(inputs.real, targets)
            imag_loss = nn.CrossEntropyLoss(inputs.imag, targets)
            return (real_loss + imag_loss)/2
        else:
            return nn.CrossEntropyLoss(inputs, targets)