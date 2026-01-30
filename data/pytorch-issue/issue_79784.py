import torch
import torch.nn as nn

loss_fn = torch.nn.MSELoss(reduction="sum")
model = nn.Sequential(
    nn.Linear(2, 2)
)

# Works on CPU
x = torch.tensor([1.0, 1.0], dtype=torch.float32)
y = torch.tensor([2.0, 2.0], dtype=torch.float32)
model.zero_grad()
y_predicted = model(x)
loss = loss_fn(y_predicted, y)
loss.backward()

# Doesn't on MPS
device = torch.device("mps")
x_mps = x.to(device=device)
y_mps = y.to(device=device)
model_mps = model.to(device)
model_mps.zero_grad()
y_predicted = model_mps(x_mps)
loss = loss_fn(y_predicted, y_mps)
loss.backward()

def test_79784():
    import torch
    import torch.nn as nn

    device = torch.device("mps")
    weight = torch.nn.Parameter(torch.randn(2, 2))


    x = torch.tensor([1.0, 1.0], dtype=torch.float32)
    out = nn.functional.linear(x, weight)
    grad = torch.autograd.grad(out.sum(), weight)
    
    x_mps = x.to(device=device)
    weight_mps = weight.to(device)
    out_mps = x_mps @ weight_mps.T
    grad_mps = torch.autograd.grad(out_mps.sum(), weight_mps)

    torch.testing.assert_close(out, out_mps, check_device=False)
    torch.testing.assert_close(grad[0], grad_mps[0], check_device=False)

    out_mps = nn.functional.linear(x_mps, weight_mps)
    grad_mps = torch.autograd.grad(out_mps.sum(), weight_mps) # fail