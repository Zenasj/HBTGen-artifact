# torch.rand(B, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, noise_std=0.2, noise_clip=0.5):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(100, 64),  # Assuming state_dim=100 and action_dim=2
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.noise_std = noise_std
        self.noise_clip = noise_clip

    def forward(self, x):
        action = self.actor(x)
        noise = (torch.randn_like(action) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
        return (action + noise).clamp(-1, 1)

def my_model_function():
    # Example hyperparameters matching user's "self.hp" (assumed values)
    return MyModel(noise_std=0.2, noise_clip=0.5)

def GetInput():
    B = 32  # Example batch size
    return torch.rand(B, 100, dtype=torch.float32)

