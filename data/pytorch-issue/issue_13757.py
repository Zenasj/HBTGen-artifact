# torch.rand(B, 80, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_inputs=80, action_dim=10, nonlinearity="relu"):
        super(MyModel, self).__init__()
        actor_dims = (256, 256)
        critic_dims = (256, 256)
        
        # Actor network layers
        self.actor_layers = nn.ModuleList()
        self.actor_layers.append(nn.Linear(num_inputs, actor_dims[0]))
        for i in range(len(actor_dims) - 1):
            self.actor_layers.append(nn.Linear(actor_dims[i], actor_dims[i+1]))
        self.mean = nn.Linear(actor_dims[-1], action_dim)
        
        # Critic network layers
        self.critic_layers = nn.ModuleList()
        self.critic_layers.append(nn.Linear(num_inputs, critic_dims[0]))
        for i in range(len(critic_dims) - 1):
            self.critic_layers.append(nn.Linear(critic_dims[i], critic_dims[i+1]))
        self.vf = nn.Linear(critic_dims[-1], 1)
        
        # Activation function selection
        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        else:
            self.nonlinearity = torch.tanh

    def forward(self, x):
        # Critic path
        critic_x = x
        for layer in self.critic_layers:
            critic_x = self.nonlinearity(layer(critic_x))
        value = self.vf(critic_x)
        
        # Actor path
        actor_x = x
        for layer in self.actor_layers:
            actor_x = self.nonlinearity(layer(actor_x))
        action = torch.tanh(self.mean(actor_x))
        
        return value, action

def my_model_function():
    # Initialize with parameters from the original example
    return MyModel(nonlinearity="relu")

def GetInput():
    # Generate random input tensor matching expected dimensions
    return torch.rand(1, 80, dtype=torch.float32)

