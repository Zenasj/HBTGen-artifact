import torch
b = torch.rand(2, 5, 5, requires_grad=True)
# for bb in b:
for i in range(len(b)):
    bb=b[i]
    bb[:, 0].clamp_(min=0, max=0.3)
    bb[:, 1].clamp_(min=0, max=0.3)

import torch

x = torch.tensor([[0.5, 0.2, 0.3, 0.8]], requires_grad=True)
w = torch.tensor([[0.2, 0.5, 0.1, 0.5]], requires_grad=True)
y_true = torch.tensor([1])

# In 1.7, `list(w)` returns list of UnbindBackward tensors, which further leads to inplace operation error.
# In previous versions, `list(w)` returns list of SelectBackward tensors and it works.
weight_list = list(w)
print('weight_list',weight_list) 

for i in range(2):
    l2_norm = torch.norm(weight_list[0], 2)
    y = w.mm(x.T)

    # mse with l2 regularization
    loss = pow(y - y_true, 2) + 0.5 * pow(l2_norm, 2)
    loss.backward()

    # simulate optimizer.step()
    with torch.no_grad():
        w.add_(w.grad, alpha=1e-3)

# weight_list = list(w)
weight_list = [w[i] for i in range(len(w))]