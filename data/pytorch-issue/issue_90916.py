import torch
from functorch import grad_and_value, vmap, make_functional_with_buffers
import torch.nn as nn


model = nn.Linear(10, 10)
fmodel, _fparams, buffers = make_functional_with_buffers(model)

criterion = nn.CrossEntropyLoss(reduction="mean")
def compute_loss_stateless_model(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch)
    loss = criterion(predictions, targets)

    return loss

ft_compute_grad = grad_and_value(compute_loss_stateless_model)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
params = list(model.parameters())

B = 256
T = 64
D = 10
inputs = torch.randn(B, D)
targets = torch.randint(0, D, (B,))

targets[1] = -100
grads, losses = ft_compute_sample_grad(params, buffers, inputs, targets)