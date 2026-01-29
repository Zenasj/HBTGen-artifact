# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)  # Example model structure
        self.params = [p for p in self.parameters() if p.requires_grad]
        self.world_size = 2  # Mock world size for demonstration

    def forward(self, x):
        # Simulate naive averaging (incorrect approach)
        params_naive = [p.clone() for p in self.params]
        for p in params_naive:
            p.data *= 2  # Mock all_reduce sum (assuming 2 processes)
            p.data /= self.world_size

        # Simulate correct averaging (flatten and allreduce)
        flat = torch.cat([p.data.reshape(-1) for p in self.params])  # Use reshape to avoid view errors
        flat *= 2  # Mock all_reduce sum
        flat /= self.world_size
        idx = 0
        params_correct = []
        for p in self.params:
            numel = p.numel()
            params_correct.append(flat[idx:idx+numel].reshape(p.shape))
            idx += numel

        # Compare results and return boolean
        differences = [torch.allclose(pn, pc) for pn, pc in zip(params_naive, params_correct)]
        return torch.tensor([all(differences)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Input tensor matching the model's expected input shape
    return torch.rand(1, 10, dtype=torch.float32)

