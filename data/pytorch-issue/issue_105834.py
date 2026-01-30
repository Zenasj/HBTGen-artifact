import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
import time
from math import pi
import numpy as np
import sobol_seq as sobol
 
class Block(nn.Module):
 

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
 
    def __init__(self, in_N, m, out_N, depth=4):
        super(drrnn, self).__init__()
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = torch.nn.Tanh()
        self.stack = torch.nn.ModuleList()
        self.stack.append(torch.nn.Linear(in_N,m))

        for i in range(depth):
            self.stack.append(Block(m,m, m))
        # last layer
        self.stack.append(torch.nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def get_interior_points(N=128):
 
    x1 = sobol.i4_sobol_generate(2, N)  
    return torch.from_numpy(x1).float() 


def get_boundary_points(N=33):
    index = sobol.i4_sobol_generate(1, N)
    xb1 = np.concatenate((index, np.zeros_like(index)), 1)
    xb2 = np.concatenate((index, np.ones_like(index)), 1)
    xb4 = np.concatenate((np.zeros_like(index), index), 1)
    xb6 = np.concatenate((np.ones_like(index), index), 1)
    xb = torch.from_numpy(np.concatenate((xb1, xb2, xb4, xb6), 0)).float()

    return xb


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def exact_sol(x):
    value = torch.where(x[:, 0: 1] > 0.5, (1 - x[:, 0: 1]) ** 2, x[:, 0: 1] ** 2)  
    return value


def function_l_exact(x):
    return x[:, 0: 1] * x[:, 1: 2] * (1 - x[:, 0: 1]) * (1 - x[:, 1: 2])


def function_f():
    return -2


def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

def error_l2(x, y):
 
    return torch.norm(x - y) / torch.norm(y)


def runmodel(epochs: int,
             lr1, lr2,
             gamma1, gamma2,
             step_size1, step_size2, N_interior, N_boundary):

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    in_N = 2
    m = 40
    out_N = 1

    device = torch.device("mps")
    soln_nn = drrnn(in_N, m, out_N).to(device) # solution
    soln_nn.apply(weights_init)
    test_function_nn = drrnn(in_N, m, out_N).to(device) # the test function
    test_function_nn.apply(weights_init)
    optimizer_solution = optim.Adam(soln_nn.parameters(), lr=lr1)
    optimizer_test_function = optim.Adam(test_function_nn.parameters(), lr=lr2)


    StepLR1 = torch.optim.lr_scheduler.StepLR(optimizer_solution, step_size=step_size1, gamma=gamma1)
    StepLR2 = torch.optim.lr_scheduler.StepLR(optimizer_test_function, step_size=step_size2, gamma=gamma2)
    tt = time.time()
    f_value = function_f()
    xr = get_interior_points(N_interior)
    xb = get_boundary_points(N_boundary)
    xr = xr.to(device)
    xb = xb.to(device)
    for epoch in range(epochs+1):

        xr.requires_grad_()
        output_r = soln_nn(xr)
        output_b = soln_nn(xb)
        output_phi_r = test_function_nn(xr) * function_l_exact(xr)
        exact_b = exact_sol(xb)
        grads_u = gradients(xr, output_r)
        grads_phi = gradients(xr, output_phi_r)

        loss_r = torch.square(torch.mean(torch.sum(grads_u * grads_phi, dim=1) - f_value * output_phi_r)) / torch.mean(torch.square(output_phi_r))
        loss_b = 10 * torch.mean(torch.abs(output_b - exact_b))
        
        loss1 = loss_r + loss_b
        loss2 = - loss_r + torch.square(torch.mean(torch.square(output_phi_r)) - 1)

        if epoch % 3 == 2:
            optimizer_test_function.zero_grad()
            loss2.backward()
            optimizer_test_function.step()
            StepLR2.step()
        else:
            optimizer_solution.zero_grad()
            loss1.backward()
            optimizer_solution.step()
            StepLR1.step()

        if epoch % 100 == 0:
            err = error_l2(soln_nn(xr), exact_sol(xr))
            print('epoch:', epoch, 'loss1:', loss1.item(), 'loss2:', loss2.item(), 'error', err.item())
            tt = time.time()

    with torch.no_grad():
        N0 = 1000
        x1 = np.linspace(0, 1, N0 + 1)

        xs1, ys1 = np.meshgrid(x1, x1)
        Z1 = torch.from_numpy(np.concatenate((xs1.flatten()[:, None], ys1.flatten()[:, None]), 1)).float()
        pred = torch.reshape(soln_nn(Z1), [N0 + 1, N0 + 1]).cpu().numpy()
        exact = torch.reshape(exact_sol(Z1), [N0 + 1, N0 + 1]).cpu().numpy()

    err = np.sqrt(np.sum(np.square(exact - pred)) / np.sum(np.square(exact)))
    print("Error:", err)

def main():
    epochs = 2000

    N_interior = 1000
    N_boundary = 200


    lr1 = 1e-2
    lr2 = 1e-2
    gamma1= 0.5
    gamma2 =0.5
    step_size1 = 1000
    step_size2 = 1000
    runmodel(epochs, lr1, lr2, gamma1, gamma2, step_size1, step_size2, N_interior, N_boundary)


if __name__ == '__main__':
    main()