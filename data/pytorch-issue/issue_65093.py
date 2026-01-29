# torch.rand(B, 1024, dtype=torch.float32)
import torch
import torch.nn as nn

class InterNodeLayerIn(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for inter-node MoE input layer with learnable parameters
        self.fc = nn.Linear(1024, 1024)
    
    def forward(self, x):
        return self.fc(x)

class IntraNodeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for intra-node MoE layer with learnable parameters
        self.fc = nn.Linear(1024, 1024)
    
    def forward(self, x):
        return self.fc(x)

class InterNodeLayerOut(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy parameter to ensure the layer stays on GPU (avoids CPU fallback)
        self.dummy_param = nn.Linear(1, 1)
    
    def forward(self, x):
        # Simulate AllToAll communication (actual implementation may vary)
        return x  # Replace with distributed operations in real use

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            InterNodeLayerIn(),
            IntraNodeLayer(),
            InterNodeLayerOut()
        )
    
    def forward(self, x):
        return self.pipeline(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input with shape inferred from logs (batch_size=8, hidden_dim=1024)
    return torch.rand(8, 1024, dtype=torch.float32)

