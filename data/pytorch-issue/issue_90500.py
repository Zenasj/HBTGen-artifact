import torch
from torch.nn.utils.stateless import functional_call
import torch.autograd as autograd
import torch.nn as nn

# This is the model
class Encoder(nn.Module):
    def __init__(self, action_dim, z_dim, skill_length):
        super().__init__()
        print(action_dim)
        self.lin1 = nn.Linear(action_dim, action_dim)
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=z_dim, batch_first=True)
        self.lin2 = nn.Linear(z_dim, z_dim)

    def forward(self, skill):
        a, b, c = skill.shape
        skill = skill.reshape(-1, skill.shape[-1])
        embed = self.lin1(skill)
        embed = embed.reshape(a, b, c)
        mean, _ = self.lstm(embed)
        mean = mean[:, -1, :]
        mean = self.lin2(mean)

        return mean

# This is the initialization function
def pars(model):
    params = {}
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            init = torch.nn.init.constant_(param, 0)
        else:
            init = torch.nn.init.orthogonal_(param)
        params[name] = nn.Parameter(init)
    return params

# Initializating the model
model = Encoder(4, 2, 5)
x = torch.rand(3, 5, 4)
params = pars(model)


# Running the model with functional_call and calculating gradient.
samp = functional_call(model, params, x)
grad_f = autograd.grad(torch.mean(samp), params.values(),
                       retain_graph=True, allow_unused=True)
print(grad_f)
# grad_f has gradient for the linear layer, but None for the LSTM layer.

# Running the model without functional_call and calculating gradient.
samp = model(x)
grad = autograd.grad(torch.mean(samp), model.parameters(), retain_graph=True)
print(grad)
# grad has gradient for all layers, e.g., linears and lstm.

nn.Linear

class Encoder(nn.Module):
    def __init__(self, action_dim, z_dim, skill_length):
        super().__init__()
        print(action_dim)
        self.lin1 = nn.Linear(action_dim, action_dim)
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=z_dim, batch_first=True)
        self.lin2 = nn.Linear(z_dim, z_dim)

    def forward(self, skill):
        a, b, c = skill.shape
        skill = skill.reshape(-1, skill.shape[-1])
        embed = self.lin1(skill)
        embed = embed.reshape(a, b, c)
        mean, _ = self.lstm(embed)
        pdb.set_trace()
        grad1 = autograd.grad(mean.mean(), params.values(),
                              retain_graph=True, allow_unused=True)
        # This gives gradient for the self.lin1 layer, and None for the LSTM
        grad2 = autograd.grad(mean.mean(), self.parameters(),
                              retain_graph=True, allow_unused=True)
        # This gives gradient the LSTM, but None for the self.lin1 layer
        mean = mean[:, -1, :]
        mean = self.lin2(mean)

        return mean