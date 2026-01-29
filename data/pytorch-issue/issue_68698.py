# torch.rand(B, C, dtype=torch.float32), torch.randint(0, C, (B,), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss().cuda()  # Matches user's CUDA setup

    def forward(self, x):
        inputs, targets = x  # Unpack input tuple
        return self.loss(inputs, targets)

def my_model_function():
    return MyModel()

def GetInput():
    B = 10  # Inferred batch size (example value)
    C = 5   # Inferred num_classes (example value)
    # Generate log probabilities (log_softmax of random logits)
    logits = torch.randn(B, C, dtype=torch.float32, device='cuda')
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    targets = torch.randint(0, C, (B,), dtype=torch.long, device='cuda')
    return (log_probs, targets)  # Tuple matches model's input requirements

