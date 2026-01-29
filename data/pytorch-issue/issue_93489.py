# torch.rand(1, 8, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.query = nn.Linear(128, 768)
        self.key = nn.Linear(128, 768)
        self.value = nn.Linear(128, 768)  # Included as in original code

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # Value layer unused in attention_scores calculation (as in original code)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        return attention_scores

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 8, 128, dtype=torch.float32)

