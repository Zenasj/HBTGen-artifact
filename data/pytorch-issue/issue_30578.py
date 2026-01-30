import torch.nn.functional as F
import random

import numpy as np
import torch
from torch.autograd import Variable


def get_numerical_gradient(func, x, epsilon):
    """
    Numerical differentiation.
    If e_ij is the matrix with 1 in row i, column j and 0 elsewhere,
    get_numerical_gradient computes
        g_ij = func(x + epsilon*e_ij) - func(x - epsilon*e_ij))/(2*epsilon)
    for all i and j and returns the gradient matrix constructed from
    the g_ij.
    """
    g = np.zeros_like(x)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            e = np.zeros_like(x)
            e[i, j] = 1.0
            g[i, j] = (func(x+epsilon*e) - func(x-epsilon*e)) / (2*epsilon)
    return g


#
# The following two functions perform the same calculation: a sum over all
# eigenvalues and eigenvectors
#
def sum_val_plus_vec_torch(x):
    d, u = torch.symeig(x, eigenvectors=True)
    return torch.sum(d) + torch.sum(u)


# This version is used for numerical differentiation.
def sum_val_plus_vec_np(x):
    x_torch = Variable(torch.from_numpy(x.copy()))
    s = sum_val_plus_vec_torch(x_torch)
    return s.data.numpy()


# for numerical gradient
epsilon = 1e-6
# random symmetric matrix
dim = 2
M = np.random.randn(dim, dim)
M = M + M.T

# numerical gradient
grad_num = get_numerical_gradient(sum_val_plus_vec_np, M, epsilon)

# torch gradient
M_torch = Variable(torch.from_numpy(M.copy()), requires_grad=True)
sum_torch = sum_val_plus_vec_torch(M_torch)
sum_torch.backward()
grad_torch = M_torch.grad.data.numpy()

print('-'*10, 'numerical differentiation')
print(grad_num)
print('-'*10, 'torch')
print(grad_torch)

M_torch = Variable(torch.from_numpy(M.copy()), requires_grad=True)
print(torch.autograd.gradcheck(sum_val_plus_vec_torch, M_torch))

class SymEig(torch.autograd.Function):
    """
    This modified version of torch.symeig has different gradients.
    The forward pass is not modified.
    """
    @staticmethod
    def forward(ctx, in_tensor):
        d, u = torch.symeig(in_tensor, eigenvectors=True)
        ctx.save_for_backward(d, u)
        return d, u

    @staticmethod
    def backward(ctx, *grad_output):
        gd, gu = grad_output
        d, u = ctx.saved_tensors
        grad_input = None

        ut = u.transpose(-2, -1)

        F = d.unsqueeze(-2) - d.unsqueeze(-1)
        F.diagonal(0, -2, -1).fill_(float('inf'))
        F.pow_(-1)

        F.mul_(torch.matmul(ut, gu))

        grad_input = torch.matmul(
            u,
            torch.matmul(torch.diag_embed(gd) + F, ut)
        )
        # until this point, no modification
        #
        # since A is symmetric, use equation (138) matrix cookbook
        return grad_input + grad_input.transpose(-2, -1) \
            - torch.diag_embed(grad_input.diagonal(dim1=-2, dim2=-1))


# test modified torch gradient
def sum_val_plus_vec_torch_modified(x):
    d, u = SymEig.apply(x)
    return torch.sum(d) + torch.sum(u)


M_torch_modified = Variable(torch.from_numpy(M.copy()), requires_grad=True)
sum_torch_modified = sum_val_plus_vec_torch_modified(M_torch_modified)
sum_torch_modified.backward()
grad_torch_modified = M_torch_modified.grad.data.numpy()

print('-'*10, 'torch modified')
print(grad_torch_modified)

def get_numerical_gradient_v(func, x, epsilon):
    assert len(x.shape)==1
    g = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        e = np.zeros(x.shape)
        e[i] = 1.0
        g[i] = (func(x+epsilon*e) - func(x-epsilon*e)) / (2*epsilon)
    return g

def symmetric_combine(v):
    n = 3.0
    M = (0.5*(v[...,None]**n + v[None,...]**n)**(1/n))
    return M

def func_numpy(v):
    M = symmetric_combine(v)
    assert(M==M.T).all()
    d, u = np.linalg.eigh(M, UPLO="U")
    result = np.sum(d) + np.sum(u)
    return result

def func_torch(v, getgrad=False):
    M = symmetric_combine(v)
    assert (M == M.T).byte().all()
    d, u = torch.symeig(M, eigenvectors=True)
    result = torch.sum(d) + torch.sum(u)
    if getgrad:
        result.backward()
        return v.grad.data.numpy()
    return result

def func_torch_modified(v, getgrad=False):
    M = symmetric_combine(v)
    assert (M==M.T).byte().all()
    d, u = SymEig.apply(M)
    result = torch.sum(d) + torch.sum(u)
    if getgrad:
        result.backward()
        return v.grad.data.numpy()
    return result

np.random.seed(2020)
# for numerical gradient
epsilon = 1e-6
# random symmetric matrix will be (dim x dim)
dim = 3
p = np.random.random(dim) + 1.0

ngrad_numpy = get_numerical_gradient_v(func_numpy, p.copy(), epsilon)
ngrad_torch = get_numerical_gradient_v(func_torch, torch.tensor(p), epsilon)
ngrad_torch_mod = get_numerical_gradient_v(func_torch_modified, torch.tensor(p), epsilon)
grad_torch = func_torch(torch.tensor(p, requires_grad=True), True)
grad_torch_mod = func_torch_modified(torch.tensor(p, requires_grad=True), True)

print('-'*10, 'numpy numerical')
print(ngrad_numpy)
print('-'*10, 'torch numerical')
print(ngrad_torch)
print('-'*10, 'torch mod numerical')
print(ngrad_torch_mod)
print('-'*10, 'torch analytic')
print(grad_torch)
print('-'*10, 'torch mod analytic')
print(grad_torch_mod)


torch.autograd.gradcheck(func_torch, torch.tensor(p, requires_grad=True))
torch.autograd.gradcheck(func_torch_modified, torch.tensor(p, requires_grad=True))

DIM = 3

def build_symmetric(M, v):
    idiag = np.diag_indices(DIM)
    iupper = np.triu_indices(DIM,1)
    M[idiag] = v[:DIM]
    M[iupper] = v[DIM:]
    M.T[iupper] = v[DIM:]
    return M

def func_numpy(v):
    M = np.empty((DIM,DIM))
    build_symmetric(M, v)
    assert(M==M.T).all()
    d, u = np.linalg.eigh(M, UPLO="U")
    result = np.sum(d) + np.sum(u)
    return result

def func_torch(v, getgrad=False):
    M = torch.empty((DIM,DIM), dtype=torch.double)
    build_symmetric(M, v)
    assert (M == M.T).byte().all()
    d, u = torch.symeig(M, eigenvectors=True)
    result = torch.sum(d) + torch.sum(u)
    if getgrad:
        result.backward()
        return v.grad.data.numpy()
    return result

def func_torch_modified(v, getgrad=False):
    M = torch.empty((DIM,DIM), dtype=torch.double)
    build_symmetric(M, v)
    assert (M==M.T).byte().all()
    d, u = SymEig.apply(M)
    result = torch.sum(d) + torch.sum(u)
    if getgrad:
        result.backward()
        return v.grad.data.numpy()
    return result

np.random.seed(2020)
# for numerical gradient
epsilon = 1e-6
p = np.random.random(int(DIM*(DIM+1)/2)) + 1.0

DIM*DIM

(DIM*(DIM+1)/2)

DIM*DIM

# copied from your code
M = torch.empty((DIM, DIM), dtype=torch.double)
M = build_symmetric(M, torch.tensor(p.copy()))
# matrix with DIM*DIM variables
MM = torch.autograd.Variable(M, requires_grad=True)
assert (MM == MM.T).byte().all()
d, u = SymEig.apply(MM)
result = torch.sum(d) + torch.sum(u)
result.backward()

print('-'*10, 'torch mod analytic')
print(MM.grad.data.numpy())