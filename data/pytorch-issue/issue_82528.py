import torch.nn as nn

3
#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import torch
import torchvision
import torch.nn.functional as F


SEED = 123
BATCH_SIZE = 64
LR = 0.01
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(SEED)
fmnist_dataset = torchvision.datasets.FashionMNIST(
    root="/tmp", train=True, download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

fmnist_dl = torch.utils.data.dataloader.DataLoader(
    fmnist_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=1, end_dim=-1),
    torch.nn.Linear(in_features=784, out_features=1000, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=1000, out_features=500, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=500, out_features=100, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=100, out_features=10, bias=True)).to(DEVICE)
loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=LR)

# Just having this dummy hook with backward(create_graph=True) produces a leak
hook_handle = model.register_full_backward_hook(lambda a, b, c: None)

for epoch in range(1, NUM_EPOCHS + 1):
    for i, (inputs, targets) in enumerate(fmnist_dl, 1):
        inputs = inputs.to(DEVICE)
        targets = torch.nn.functional.one_hot(
            targets, num_classes=10).to(inputs.dtype).to(DEVICE)
        #
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward(create_graph=True)
        opt.step()

        # removing the hook fixes the leak...
        hook_handle.remove()
        # ... unless we create a new hook!
        hook_handle = model.register_full_backward_hook(lambda a, b, c: None)

        # Also note that this suggested approach doesn't help
        for param in model.parameters():
            param.grad = None

        #
        if i % 50 == 0:
            print(f"[{epoch}/{i}]:", loss.item())

3
HANDLES = []

3
handle = grad_fn.register_hook(hook)
global HANDLES
HANDLES.append(handle)

3
for epoch in range(1, NUM_EPOCHS + 1):
    for i, (inputs, targets) in enumerate(fmnist_dl, 1):
        inputs = inputs.to(DEVICE)
        targets = torch.nn.functional.one_hot(
            targets, num_classes=10).to(inputs.dtype).to(DEVICE)
        #
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward(create_graph=True)
        opt.step()
        #
        if i % 50 == 0:
            print(f"[{epoch}/{i}]:", loss.item())
        # these are the lines to hotfix the leak
        while torch.utils.hooks.HANDLES:
            xx = torch.utils.hooks.HANDLES.pop()
            xx.remove()

from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
import torch

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader

download_root = './MNIST_DATASET'
import torchvision.transforms as transforms
import functools



# Normalize data with mean=0.5, std=1.0
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
# option 값 정의
batch_size = 512
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)
device = torch.device("cuda:0")

def compute_distance_grads_var(dict_grad_1,dict_grad_2):
    penalty = 0
    penalty += l2_between_lists(dict_grad_1, dict_grad_2)
    return penalty

def l2_between_lists(list_1, list_2):
    assert len(list_1) == len(list_2)
    return (
        torch.cat(tuple([t.view(-1) for t in list_1])) -
        torch.cat(tuple([t.view(-1) for t in list_2]))
    ).pow(2).sum()

def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /(torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
    elif method == 'l2_mean':
        dist_ = torch.norm(x-y, 2)
    return dist_

def get_grads(logits, y,model,bce_extended,real_sample):
    loss = bce_extended(logits, y).sum()
    with backpack(BatchGrad()):
        if real_sample:
            torch.autograd.grad(loss, list(model.parameters()))
        else:
            torch.autograd.grad(loss, list(model.parameters()),create_graph = True)

    grads_mean = []
    dict_grads_batch = []

    for name, weights in model.named_parameters():
        if real_sample:
            # grads_mean.append(weights.grad.detach().clone())
            dict_grads_batch.append(weights.grad_batch.detach().clone().view(weights.grad_batch.size(0), -1))
        else:
            # grads_mean.append(weights.grad)
            dict_grads_batch.append(weights.grad_batch.clone().view(weights.grad_batch.clone().size(0), -1))

    return grads_mean, dict_grads_batch


for i in range(100):
    model = Sequential(Flatten(), Linear(784, 128), Linear(128, 10))  # I added an additional layer here
    lossfunc = CrossEntropyLoss()
    model = extend(model)
    lossfunc = extend(lossfunc)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (x, target) in enumerate(train_loader):
        x_syn = torch.rand(x.shape, requires_grad=True, device="cuda:0")
        y_syn = torch.ones_like(target)
        y_syn = y_syn.to(device)
        optimizer_alpha = torch.optim.Adam([x_syn], lr=1e-3)

        x = x.to(device)
        target = target.to(device)
        grad, grad_batch = get_grads(model(x), target, model, lossfunc,real_sample=True)
        optimizer.zero_grad()
        grad_syn, grad_batch_syn = get_grads(model(x_syn), y_syn, model, lossfunc,real_sample=False)
        loss =0
        loss += compute_distance_grads_var(grad_batch,grad_batch_syn)
        loss.backward()
        optimizer_alpha.step()
        optimizer_alpha.zero_grad()