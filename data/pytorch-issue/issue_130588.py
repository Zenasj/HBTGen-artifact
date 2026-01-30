import torch
import torch.library
import torch.nn as nn
import torch.nn.functional as F

# --- Define a custom Op using torch.library --- #
lib = torch.library.Library("example", "DEF")
lib.define("custom_op(Tensor x) -> Tensor")

def custom_op(x: torch.Tensor):
    return x

def custom_op_fake(x: torch.Tensor):
    return x

lib.impl("custom_op", custom_op, "CPU")  
torch.library.register_fake("example::custom_op", custom_op_fake)

# --- Create a Model with the custom Op ---

# 1 input with multiple (debug outputs).
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        
    def forward(self, x):
        y_fc1 = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y_fc1))
        return torch.ops.example.custom_op(y)

if __name__ == "__main__":

    # generate a uniform distribution of data.
    n_mini_batches = 10
    x_in = torch.distributions.uniform.Uniform(-1, 1).sample([n_mini_batches, 8, 64])

    # PRIME 1xN (debug)
    X = (x_in[1,:],)

    # export the model.
    m_export = torch.export.export(MLP().eval(), X).module()