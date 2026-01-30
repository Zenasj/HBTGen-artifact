import torch

x = torch.tensor([16., 0.], requires_grad=True)
y = x/2  # tensor([8., 0.], grad_fn=<DivBackward0>)
z = x.sqrt() + 1  # tensor([5., 1.], grad_fn=<SqrtBackward>)

# Calculate dy/dx, dz/dx
dydx = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]  # tensor([0.5000, 0.5000])
dzdx = torch.autograd.grad(z.sum(), x, retain_graph=True)[0]  # tensor([0.1250,    inf])

# Define w = [w0, w1] == [y0, z1]
w = torch.where(x == 0., y, z)  # tensor([5., 0.], grad_fn=<SWhereBackward>)
expected_dw_dx = torch.where(x == 0., dydx, dzdx)  # tensor([0.1250, 0.5000])
dwdx = torch.autograd.grad(w.sum(), x, retain_graph=True)[0]  # is actually tensor([0.1250, inf])
print("`torch.where` communicates gradients correctly:", torch.equal(expected_dw_dx, dwdx))

x = torch.tensor([-1., 1.], requires_grad=True)
y = 2 * x  # tensor([-2., 2.], grad_fn=<MulBackward0>)
z = 3 * x  # tensor([-3., 3.], grad_fn=<MulBackward0>)

# Calculate dy/dx, dz/dx
dydx = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]  # tensor([2., 2.])
dzdx = torch.autograd.grad(z.sum(), x, retain_graph=True)[0]  # tensor([3., 3.])

# Define w = [w0, w1] == [y0, z1]
w = torch.where(x < 0., y, z)  # tensor([-2.,  3.], grad_fn=<SWhereBackward>)
expected_dw_dx = torch.where(x < 0., dydx, dzdx)  # tensor([-2.,  3.])
dwdx = torch.autograd.grad(w.sum(), x, retain_graph=True)[0]  # is correctly tensor([2., 3.])

# Check
print("`torch.where` communicates gradients correctly:", torch.equal(expected_dw_dx, dwdx))  # True