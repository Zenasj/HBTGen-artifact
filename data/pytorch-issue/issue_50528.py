# torch.randn(B, 10), torch.randint(10, (B,))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.criterion_mean = nn.CrossEntropyLoss(weight=self.weight, reduction='mean')
        self.criterion_none = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        self.criterion_no_weight = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs):
        logits, target = inputs
        batch_size = logits.shape[0]

        # Check reduction inconsistency
        loss_mean = self.criterion_mean(logits, target)
        loss_none = self.criterion_none(logits, target).mean()
        diff_reductions = not torch.allclose(loss_mean, loss_none)

        # Check batch_size=1 weight issue
        diff_weight = False
        if batch_size == 1:
            loss_with = self.criterion_mean(logits, target)
            loss_without = self.criterion_no_weight(logits, target)
            diff_weight = torch.allclose(loss_with, loss_without)

        # Return True if any discrepancy found
        return torch.tensor(diff_reductions or (batch_size == 1 and diff_weight), dtype=torch.bool)

def my_model_function():
    num_classes = 10
    weight = torch.rand(num_classes)  # Matches the 10-class scenario in the issue
    return MyModel(weight)

def GetInput():
    batch_size = torch.randint(1, 9, (1,)).item()  # Random batch size 1-8 (including edge case)
    num_classes = 10
    logits = torch.randn(batch_size, num_classes)
    target = torch.randint(num_classes, (batch_size,))
    return (logits, target)

