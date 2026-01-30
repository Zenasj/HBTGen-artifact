import torch

with autograd.detect_anomaly():
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    grad_sum = 0
    for grad in grads:
        grad_sum += 0.1 * grad.pow(2).sum()

    grad_sum.backward(retain_graph=False)