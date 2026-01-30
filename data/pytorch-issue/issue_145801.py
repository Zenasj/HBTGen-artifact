import torch.nn as nn

import torch

t = torch.load('fc_layer_tensor.pt', weights_only=True, map_location='cpu').flatten()
torch.linalg.eigh(torch.outer(t, t))

import torch
from torchvision import datasets, transforms

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 512
num_classes = 10
num_features = 28**2
loss_fn = torch.nn.CrossEntropyLoss()

tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST("~/data/", download=False, train=True, transform=tforms)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

fc_layer = torch.nn.Linear(in_features=num_features, out_features=num_classes, bias=False).to(DEVICE)

for batch_ix, (inputs, targets) in enumerate(train_loader):

    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    fc_layer.weight.grad = None
    logits = fc_layer(inputs.view(inputs.shape[0], -1))
    loss = loss_fn(logits, targets)
    loss.backward()

    vec_grad = torch.flatten(fc_layer.weight.grad)
    precond_adagrad = torch.outer(vec_grad, vec_grad)

    # CPU computation works fine
    evals_adagrad, evecs_adagrad = torch.linalg.eigh(precond_adagrad.cpu())

    # But eigh computation on GPU fails
    evals_adagrad, evecs_adagrad = torch.linalg.eigh(precond_adagrad)