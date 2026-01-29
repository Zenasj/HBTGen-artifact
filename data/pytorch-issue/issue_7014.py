# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, temperature=1.0, max_logit=40.0):
        super().__init__()
        self.temperature = temperature
        self.max_logit = max_logit

    def forward(self, logits):
        # Clamp logits to prevent overflow
        clamped = logits.clamp(max=self.max_logit)
        # Normalize by subtracting max for numerical stability
        normalized = clamped - clamped.max(dim=-1, keepdim=True)[0]
        scaled = normalized / self.temperature
        probs = torch.exp(scaled)
        return torch.multinomial(probs, 1)

def my_model_function():
    return MyModel(temperature=1.0, max_logit=40.0)

def GetInput():
    B, C = 1, 3  # Batch size and number of classes
    input_tensor = torch.rand(B, C, dtype=torch.float32)
    # Introduce an inf in one element to test CUDA handling
    input_tensor[0, 0] = float('inf')
    return input_tensor.cuda()  # Explicitly test CUDA execution

