# torch.randn(1,5,10), torch.full((1,5), -100, dtype=torch.long) ← inferred input shapes for (logits, labels)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.loss_fct_mean = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, inputs):
        logits, labels = inputs
        # Compute both loss reductions
        loss_none = self.loss_fct(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )
        loss_mean = self.loss_fct_mean(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )
        
        # Check if none-reduction is all zeros and mean is NaN
        is_loss_none_zero = torch.allclose(loss_none, torch.zeros_like(loss_none))
        is_loss_mean_nan = torch.isnan(loss_mean)
        
        return torch.tensor(is_loss_none_zero and is_loss_mean_nan, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    B, L, C = 1, 5, 10
    logits = torch.randn(B, L, C)  # Matches issue's input shape
    labels = torch.full((B, L), -100, dtype=torch.long)  # All ignored indices
    return (logits, labels)

