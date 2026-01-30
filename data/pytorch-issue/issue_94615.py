import torch.nn as nn

from matplotlib import cm
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import numpy as np

class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        # Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()  # For second-order optimizers important
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        # Apply update step to all parameters
        for p in self.params:
            if p.grad is None:  # We skip parameters without any gradients
                continue
            self.update_param(p)

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError

class SGDMomentum(OptimizerTemplate):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum  # Corresponds to beta_1 in the equation above
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}  # Dict to store m_t

    def update_param(self, p):
        self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
        p_update = -self.lr * self.param_momentum[p]
        p.add_(p_update)

def pathological_curve_loss(w1, w2):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x1_loss = torch.tanh(w1) ** 2 + 0.01 * torch.abs(w1)
    x2_loss = torch.sigmoid(w2)
    return x1_loss + x2_loss

def plot_curve(
    curve_fn, x_range=(-5, 5), y_range=(-5, 5), plot_3d=False, cmap=cm.viridis, title="Pathological curvature"
):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') if plot_3d else fig.gca()
#     ax = fig.gca(projection="3d") if plot_3d else fig.gca()

    x = torch.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    y = torch.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / 100.0)
    x, y = torch.meshgrid([x, y])
    z = curve_fn(x, y)
    x, y, z = x.numpy(), y.numpy(), z.numpy()

    if plot_3d:
        ax.plot_surface(x, y, z, cmap=cmap, linewidth=1, color="#000", antialiased=False)
        ax.set_zlabel("loss")
    else:
        ax.imshow(z.T[::-1], cmap=cmap, extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.title(title)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    plt.tight_layout()
    return ax

def train_curve(optimizer_func, curve_func=pathological_curve_loss, num_updates=100, init=[5, 5]):
    """
    Args:
        optimizer_func: Constructor of the optimizer to use. Should only take a parameter list
        curve_func: Loss function (e.g. pathological curvature)
        num_updates: Number of updates/steps to take when optimizing
        init: Initial values of parameters. Must be a list/tuple with two elements representing w_1 and w_2
    Returns:
        Numpy array of shape [num_updates, 3] with [t,:2] being the parameter values at step t, and [t,2] the loss at t.
    """
    weights = nn.Parameter(torch.FloatTensor(init), requires_grad=True)
    optim = optimizer_func([weights])

    list_points = []
    for _ in range(num_updates):
        loss = curve_func(weights[0], weights[1])
        list_points.append(torch.cat([weights.data.detach(), loss.unsqueeze(dim=0).detach()], dim=0))
        optim.zero_grad()
        loss.backward()
        optim.step()
    points = torch.stack(list_points, dim=0).numpy()
    return points


# BEGIN only place changed from https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
SGDMom_points = train_curve(lambda params: torch.optim.SGD(params, lr=10, momentum=0.9))
# END only place changed from https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
SGDMom_tutorial_points = train_curve(lambda params: SGDMomentum(params, lr=10, momentum=0.9))


all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
ax = plot_curve(
    pathological_curve_loss,
    x_range=(-np.absolute(all_points[:, 0]).max(), np.absolute(all_points[:, 0]).max()),
    y_range=(all_points[:, 1].min(), all_points[:, 1].max()),
    plot_3d=False,
)
ax.plot(SGDMom_points[:, 0], SGDMom_points[:, 1], color="red", marker="o", zorder=2, label="SGDMom from torch")
ax.plot(SGDMom_tutorial_points[:, 0], SGDMom_tutorial_points[:, 1], color="blue", marker="o", zorder=2, label="SGDMom from scratch")
plt.legend()
plt.show()

import torch
import matplotlib.pyplot as plt

def pathological_curve_loss(w1, w2):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x1_loss = torch.tanh(w1) ** 2 + 0.01 * torch.abs(w1)
    x2_loss = torch.sigmoid(w2)
    return x1_loss + x2_loss

num_updates=100
weights = torch.nn.Parameter(torch.FloatTensor([5,5]),requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=10, momentum=0.9)
list_points=[]
for _ in range(num_updates):
    loss = pathological_curve_loss(weights[0],weights[1])
    list_points.append(torch.cat([weights.data.detach(), loss.unsqueeze(dim=0).detach()], dim=0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
points = torch.stack(list_points, dim=0).numpy()
    
plt.plot(points[:, 0], points[:, 1], color="green", marker="o", zorder=2, label="SGDMom from torch")
plt.legend()
plt.show()

class SGDMomentum(OptimizerTemplate):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum  # Corresponds to beta_1 in the equation above
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}  # Dict to store m_t

    def update_param(self, p):
#         self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
        self.param_momentum[p] = p.grad + self.momentum * self.param_momentum[p]
        p_update = -self.lr * self.param_momentum[p]
        p.add_(p_update)