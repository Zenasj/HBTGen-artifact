# torch.rand(B, 12, 2048, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_cxt=5):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2048, 1)  # Compute attention weights
        self.num_cxt = num_cxt

    def forward(self, value):
        # Compute weights (B, 12, 2048) -> (B,12)
        weights = self.fc(value).squeeze(-1)
        # Get topK indices along dim=1
        _, idx = torch.topk(weights, self.num_cxt, dim=1)
        # Expand indices to match value dimensions
        batch_size = value.size(0)
        idx_expand = idx.unsqueeze(-1).expand(
            batch_size, self.num_cxt, value.size(2)
        )
        # Gather values using expanded indices
        candidate_value = torch.gather(value, dim=1, index=idx_expand.long())
        return candidate_value

def my_model_function():
    return MyModel(num_cxt=5)  # Matches the 5-element indices in examples

def GetInput():
    # Matches value_size dimensions [B, 12, 2048]
    return torch.rand(2, 12, 2048, dtype=torch.float32)

