import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

seed = 142


class Square(nn.Module):

    def __init__(self):
        super(Square, self).__init__()
        self.a = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (self.a * x)**2


def grad_of_torch_func_backward(z, func):
    with torch.enable_grad():
        z.requires_grad_(True)
        omega = func(z).reshape(z.size(0), 1).sum()
        torch.autograd.backward(omega)
        dz = z.grad
        dz.requires_grad_(True)
        return dz


def grad_of_torch_func(z, func):
    with torch.enable_grad():
        z.requires_grad_(True)
        omega = func(z).reshape(z.size(0), 1)
        go = torch.ones_like(omega).to(z.device)
        dz = torch.autograd.grad(outputs=omega,
                                 inputs=z,
                                 grad_outputs=go,
                                 create_graph=True,
                                 retain_graph=True)[0]
        return dz


def objective_wrong_grad(model: nn.Module, y: torch.Tensor):
    y.requires_grad_(True)
    g = grad_of_torch_func_backward(y, model)
    y.requires_grad_(False)
    return y - g


def objective_correct_grad(model: nn.Module, y: torch.Tensor):
    y.requires_grad_(True)
    g = grad_of_torch_func(y, model)
    y.requires_grad_(False)
    return y - g


def correct_param_autograd():
    torch.manual_seed(seed)
    se = Square()

    x = torch.ones(10, 1)
    y = x + 0.1 * torch.randn_like(x)
    z = objective_correct_grad(se, y)

    loss = cp.checkpoint(F.mse_loss, z, x)
    loss.backward()

    param_grad_correct = se.a.grad.item()
    param = se.a.item()

    return x.data.numpy(), y.data.numpy(), z.data.numpy(), loss.item(), param, param_grad_correct


def incorrect_param_autograd():
    torch.manual_seed(seed)
    se = Square()

    x = torch.ones(10, 1)
    y = x + 0.1 * torch.randn_like(x)
    z = objective_wrong_grad(se, y)

    loss = F.mse_loss(z, x)
    loss.backward()

    param_grad_incorrect = se.a.grad.item()
    param = se.a.item()

    return x.data.numpy(), y.data.numpy(), z.data.numpy(), loss.item(), param, param_grad_incorrect


def compute_analytic_param_grad(a, x, y):
    return (8 * a * y * (x + (2 * a**2 - 1) * y)).mean().item()


def main():
    x0, y0, z0, loss0, p0, p_grad0 = correct_param_autograd()
    x1, y1, z1, loss1, p1, p_grad1 = incorrect_param_autograd()

    np.testing.assert_almost_equal(p0, p1)
    np.testing.assert_array_almost_equal(x0, x1)
    np.testing.assert_array_almost_equal(y0, y1)
    np.testing.assert_array_almost_equal(z0, z1)
    np.testing.assert_almost_equal(loss0, loss1)

    analytic_grad = compute_analytic_param_grad(p0, x0, y0)
    np.testing.assert_almost_equal(p_grad0, analytic_grad)

    np.testing.assert_almost_equal(p_grad0, p_grad1)


if __name__ == '__main__':
    main()