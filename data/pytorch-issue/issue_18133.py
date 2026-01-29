# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.distributions as dist

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Neural network head to generate distribution parameters
        self.fc1 = nn.Linear(3*4*5, 100)  # Input shape (C=3, H=4, W=5)
        self.categorical_logits = nn.Linear(100, 3)  # 3-class Categorical
        self.normal_params = nn.Linear(100, 2)  # loc and scale for Normal
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        
        # Generate parameters for both distributions
        cat_logits = self.categorical_logits(x)
        norm_params = self.normal_params(x)
        norm_loc, norm_scale = norm_params.split(1, dim=1)
        norm_scale = norm_scale.squeeze(-1)
        
        error_flag = 0
        try:
            # Attempt to create distributions (triggers constraint checks)
            _ = dist.Categorical(logits=cat_logits)
            _ = dist.Normal(norm_loc, norm_scale)
        except ValueError:
            error_flag = 1  # Flag invalid parameters
            
        return torch.tensor([error_flag], dtype=torch.long)

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape compatible with MyModel's forward pass
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

