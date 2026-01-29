# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.kld_loss = nn.KLDivLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input):
        # Generate target tensor of the same shape as input
        target = torch.zeros_like(input)
        # BCELoss expects probabilities between 0 and 1
        input_bce = torch.sigmoid(input)
        # KLDivLoss expects log probabilities along the channel dimension
        input_kld = torch.log_softmax(input, dim=1)
        # Compute losses
        mse_out = self.mse_loss(input, target)
        kld_out = self.kld_loss(input_kld, target)
        bce_out = self.bce_loss(input_bce, target)
        return mse_out, kld_out, bce_out

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor with requires_grad=True to avoid the set_requires_grad error
    return torch.rand(2, 3, 4, 5, dtype=torch.float32, requires_grad=True)

