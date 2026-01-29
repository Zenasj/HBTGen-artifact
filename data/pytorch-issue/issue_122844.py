# torch.rand(B, 5, dtype=torch.float32)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock SB3 policy network (input size inferred from dummy_input: 5 features)
        self.policy = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Assuming action space has 2 outputs
        )
    
    def forward(self, observation):
        # SB3 policy returns (actions, values, log_probs)
        # Mock outputs for ONNX compatibility
        actions = self.policy(observation)
        values = torch.tensor([0.0], device=observation.device)  # Dummy value
        log_probs = torch.tensor([0.0], device=observation.device)  # Dummy log probability
        return actions, values, log_probs  # Must return all 3 outputs as in original policy

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the dummy_input from the issue
    distance = np.linalg.norm(np.array([0, 0]) - np.array([1, 1]))
    return torch.tensor([[0.0, 0.0, 1.0, 1.0, distance]], dtype=torch.float32)

