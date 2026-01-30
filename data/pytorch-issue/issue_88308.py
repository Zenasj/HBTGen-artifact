import torch.nn as nn

import torch
from functorch import vmap, grad, jacrev, make_functional

device =  "mps" if torch.backends.mps.is_available() else "cpu" #change to cpu to reproduce cpu output 

class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 1),
            )
        
    def forward(self, x):
        out =  self.net(x)
        return out
        
dnn = DNN().to(device)
fmodel, params = make_functional(dnn)

batch_size = 3
data = torch.ones(batch_size, 1).to(device)
targets = torch.ones(batch_size, 1).to(device)

#Modified from various functorch tutorials: 
def loss_fn(predictions, targets):
    return torch.nn.functional.mse_loss(predictions, targets)

def compute_loss_stateless_model (params, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, batch) 
    loss = loss_fn(predictions, targets)
    return loss

def comp_loss (params, data, target):
    predictions = fmodel(params, data) 
    loss = loss_fn(predictions, targets)
    return loss    

ft_compute_grad = grad(compute_loss_stateless_model)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
grads = ft_compute_sample_grad(params, data, targets)
compute_sample_hess = jacrev(vmap(ft_compute_grad, in_dims=(None, 0, 0)))

hess = compute_sample_hess(params,data,targets)
print(hess)

import torch
from functorch import vmap, grad, jacrev, make_functional

device =  "mps" if torch.backends.mps.is_available() else "cpu" #change to cpu to reproduce cpu output 

class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 2), 
            torch.nn.Tanh(),
            torch.nn.Linear(2, 1),
            )
        
    def forward(self, x):
        out =  self.net(x)
        return out
        
dnn = DNN()
hessians = []
for device in ['mps', 'cpu']:
    dnn.to(device)
    fmodel, params = make_functional(dnn)
    
    batch_size = 3
    data = torch.ones(batch_size, 1).to(device)
    targets = torch.ones(batch_size, 1).to(device)
    
    #Modified from various functorch tutorials: 
    def loss_fn(predictions, targets):
        return torch.nn.functional.mse_loss(predictions, targets)
    
    def compute_loss_stateless_model (params, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
    
        predictions = fmodel(params, batch) 
        loss = loss_fn(predictions, targets)
        return loss
    
    def comp_loss (params, data, target):
        predictions = fmodel(params, data) 
        loss = loss_fn(predictions, targets)
        return loss    
    
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
    grads = ft_compute_sample_grad(params, data, targets)
    compute_sample_hess = jacrev(vmap(ft_compute_grad, in_dims=(None, 0, 0)))
    
    hess = compute_sample_hess(params,data,targets)
    hessians.append(hess)


for a, b in zip(hessians[0], hessians[1]):
    if isinstance(a, tuple):
        for _a, _b in zip(a,b):
            torch.testing.assert_close(_a.to('cpu'), _b)
    else: 
        torch.testing.assert_close(a.to('cpu'), b)