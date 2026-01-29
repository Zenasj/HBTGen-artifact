# torch.rand(32, 2, 177, dtype=torch.float32).cuda()  # Inferred input shape and dtype
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True).cuda()
        # Dummy tensors matching shapes reported in issue comments
        self.targets = torch.randint(0, 177, (19,), dtype=torch.int32).cuda()  # Targets must be on same device as log_probs
        self.input_lengths = torch.full((2,), 32, dtype=torch.int32).cuda()   # T=32 for all samples
        self.target_lengths = torch.tensor([10, 9], dtype=torch.int32).cuda() # Sum to 19 (matches targets.shape[0])

    def forward(self, log_probs):
        return self.ctc_loss(log_probs, self.targets, self.input_lengths, self.target_lengths)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 2, 177, dtype=torch.float32).cuda()

