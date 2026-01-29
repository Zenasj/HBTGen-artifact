# torch.randint(0, 38, (B,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.onehot_dim = 38  # From _char_to_onehot parameter

    def forward(self, input_char):
        return self._char_to_onehot(input_char, self.onehot_dim)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        device = input_char.device  # Use input's device
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

def my_model_function():
    return MyModel()

def GetInput():
    B = 5  # Example batch size; can be adjusted
    return torch.randint(0, 38, (B,), dtype=torch.long)

