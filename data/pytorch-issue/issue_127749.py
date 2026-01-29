# torch.rand(4, 400, 256, dtype=torch.float32)  # Inferred input shape from the issue
import torch.nn.functional as F
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.num_vars = 320

        self.weight_proj = nn.Linear(256, self.num_groups * self.num_vars)
        self.temperature = 2

        self.weight_proj.weight.data.normal_(mean=0.0, std=1)
        self.weight_proj.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        codevector_probs = F.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)
        return codevector_probs

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, 400, 256, dtype=torch.float32).cuda()

