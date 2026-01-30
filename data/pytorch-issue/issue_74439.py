S_diff = torch.sign((labels[None, :] - labels[:, None]).squeeze().T)
S_diff = torch.sign((labels[:, None] - labels[None, :]).squeeze())

import torch
import torch.nn as nn
import copy

# Loss version 1
def compute_lambda_i_version1(scores, labels):
    scores = scores.squeeze(1)
    z_diff = (scores[:, None] - scores[None, :]).squeeze()
    S_diff = torch.sign((labels[None, :] - labels[:, None]).squeeze().T) # Difference

    lambda_i =  (1 - S_diff)  / 2 - 1 /(1 + torch.exp(z_diff))
    lambda_i = lambda_i.sum(axis=1).unsqueeze(1)
        
    return lambda_i 

# Loss version 2
def compute_lambda_i_version2(scores, labels):
    scores = scores.squeeze(1)
    z_diff = (scores[:, None] - scores[None, :]).squeeze()
    S_diff = torch.sign((labels[:, None] - labels[None, :]).squeeze()) # Difference

    lambda_i =  (1 - S_diff)  / 2 - 1 /(1 + torch.exp(z_diff))

    lambda_i = lambda_i.sum(axis=1).unsqueeze(1)
        
    return lambda_i 

# Generic neural net
class NeuralModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                        nn.Linear(501,256),
                        nn.ReLU(),
                        nn.Linear(256,1)
                        )
    
    def forward(self, x):
        x = self.network(x)
        return x

# Random input
inputs = torch.rand(5, 501)
labels = torch.FloatTensor([1, 2, 3, 0, 4])

# Init the networks
net_1 = NeuralModule()
net_2 = NeuralModule()
net_2.load_state_dict(copy.deepcopy(net_1.state_dict()))

# Network 1 gradients / loss version 1
scores_1 = net_1(inputs)
loss_1 = compute_lambda_i_version1(scores_1, labels)
torch.autograd.backward(scores_1, loss_1)
grad_1 = net_1.network[2].weight.grad

# Network 2 gradients / loss version 2
scores_2 = net_2(inputs)
loss_2 = compute_lambda_i_version2(scores_2, labels)
torch.autograd.backward(scores_2, loss_2)
grad_2 = net_2.network[2].weight.grad

print(torch.equal(grad_1, grad_2))
print(torch.abs(grad_1-grad_2).mean())

print(torch.equal(loss_1, loss_2))
print(torch.abs(loss_1-loss_2).mean())