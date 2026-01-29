# torch.randint(0, 2, (1,), dtype=torch.long)  # Input shape: (num_features,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_features=1, threshold=1e-5, min_observation=100):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.min_observation = min_observation
        # Track counters for both strategies
        self.register_buffer('old_mismatched', torch.zeros(num_features, dtype=torch.long))
        self.register_buffer('old_total', torch.zeros(num_features, dtype=torch.long))
        self.register_buffer('new_mismatched', torch.zeros(num_features, dtype=torch.long))
        self.register_buffer('new_total', torch.zeros(num_features, dtype=torch.long))
    
    def forward(self, input):
        diff = torch.zeros(self.num_features, dtype=torch.long)
        for i in range(self.num_features):
            # Old strategy logic (cumulative counters)
            self.old_mismatched[i] += input[i].item()
            self.old_total[i] += 1
            old_ratio = self.old_mismatched[i].float() / self.old_total[i].float()
            old_exception = (self.old_total[i] >= self.min_observation) and (old_ratio > self.threshold)
            
            # New strategy logic (reset counters on exception)
            self.new_mismatched[i] += input[i].item()
            self.new_total[i] += 1
            new_ratio = self.new_mismatched[i].float() / self.new_total[i].float()
            new_exception = (self.new_total[i] >= self.min_observation) and (new_ratio > self.threshold)
            
            # Reset new counters if exception triggered
            if new_exception:
                self.new_mismatched[i] = 0
                self.new_total[i] = 0
            
            # Track differences in exception triggering behavior
            diff[i] = 1 if old_exception != new_exception else 0
        return diff

def my_model_function():
    # Initialize with parameters matching the test scenario
    return MyModel(num_features=1, threshold=1e-5, min_observation=100)

def GetInput():
    # Simulate mismatch count (0/1) for a single feature per batch
    return torch.randint(0, 2, (1,), dtype=torch.long)

