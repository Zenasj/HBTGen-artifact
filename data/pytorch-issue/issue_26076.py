import torch

def test_grad():
    x1 = torch.tensor([[0.]], requires_grad=True)
    x2 = torch.tensor([[0.5], [1.0]], requires_grad=True)
    res = torch.cdist(x1, x2)
    res[0, 0].backward()
    print(x1.grad, x2.grad)

x1 = torch.tensor([[0.]], requires_grad=True)
x2 = torch.tensor([[0.5], [1.0]], requires_grad=True)