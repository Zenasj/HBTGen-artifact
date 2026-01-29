# torch.rand(50, 3, 15, dtype=torch.float64), targets=torch.randint(0, 2, (3, 30)), input_lengths=torch.tensor([50, 50, 50], dtype=torch.int32), target_lengths=torch.tensor([30, 25, 20], dtype=torch.int32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blank = 14
        self.reduction = "mean"
        self.zero_infinity = False

    def forward(self, inputs):
        log_probs, targets, input_lengths, target_lengths = inputs
        return F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

def my_model_function():
    return MyModel()

def GetInput():
    log_probs = torch.rand(50, 3, 15, dtype=torch.float64)
    targets = torch.randint(0, 2, (3, 30), dtype=torch.int64)
    input_lengths = torch.tensor([50, 50, 50], dtype=torch.int32)
    target_lengths = torch.tensor([30, 25, 20], dtype=torch.int32)
    return (log_probs, targets, input_lengths, target_lengths)

