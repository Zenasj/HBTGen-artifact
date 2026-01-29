# torch.rand(B, 501, dtype=torch.float32) and labels of shape (B,)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(501, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        inputs, labels = x
        scores = self.network(inputs).squeeze(1)
        
        # Compute loss version1
        z_diff1 = (scores[:, None] - scores[None, :]).squeeze()
        S_diff1 = torch.sign((labels[None, :] - labels[:, None]).squeeze().T)
        lambda_i1 = (1 - S_diff1) / 2 - 1 / (1 + torch.exp(z_diff1))
        loss1 = lambda_i1.sum(axis=1).unsqueeze(1)
        
        # Compute loss version2
        z_diff2 = (scores[:, None] - scores[None, :]).squeeze()
        S_diff2 = torch.sign((labels[:, None] - labels[None, :]).squeeze())
        lambda_i2 = (1 - S_diff2) / 2 - 1 / (1 + torch.exp(z_diff2))
        loss2 = lambda_i2.sum(axis=1).unsqueeze(1)
        
        # Return mean absolute difference between the two losses
        return torch.abs(loss1 - loss2).mean()

def my_model_function():
    return MyModel()

def GetInput():
    B = 5  # Batch size from the original example
    inputs = torch.rand(B, 501, dtype=torch.float32)
    labels = torch.tensor([1., 2., 3., 0., 4.], dtype=torch.float32)  # Original labels
    return (inputs, labels)

