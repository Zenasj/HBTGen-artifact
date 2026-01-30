import torch.nn as nn

class xxx(nn.Module):
    def forward(xxx):
        xxx = self.cost_memory_function(xxx)
        ...  # OOM error occurs here rather than at the above function.
        return xxx
    def cost_memory_function(xxx):
        ...