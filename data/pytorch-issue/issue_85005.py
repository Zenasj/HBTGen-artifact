# torch.rand(B, C, dtype=torch.float32, device='cuda')  # e.g., B=65536, C=65537

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()  # Core loss module

    def forward(self, logits):
        # Workaround for CUDA illegal memory access with large batches
        batch_size = logits.size(0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        half = batch_size // 2
        loss1 = self.ce(logits[:half], labels[:half])
        loss2 = self.ce(logits[half:], labels[half:])
        return (loss1 + loss2) / 2  # Average loss of two halves

def my_model_function():
    return MyModel()

def GetInput():
    B = 2**16  # Batch size causing CUDA error in original issue
    C = 2**16 + 1  # Number of classes
    return torch.rand(B, C, dtype=torch.float32, device='cuda')

