import random

x = torch.sparse.FloatTensor(5, 5)
y = torch.FloatTensor(5, 5)

torch.mm(x, y) # works

xx = torch.autograd.Variable(x)
xy = torch.autograd.Variable(y)

torch.mm(x, y) # fails

x = torch.sparse.FloatTensor(5, 5)
y = torch.FloatTensor(5, 5)

torch.mm(x, y) # works

xx = torch.autograd.Variable(x)
xy = torch.autograd.Variable(y)
print(type(x), type(xx))

torch.mm(xx, xy) # I think here we want to check if backward things

x = torch.sparse.FloatTensor(5, 5)
y = torch.FloatTensor(5, 5)
xx = torch.autograd.Variable(x)
xy = torch.autograd.Variable(y)
print(type(x), type(xx))
torch.mm(xx, xy) # I think here we want to check if backward things

x = torch.sparse.FloatTensor(5, 5)
y = torch.sparse.FloatTensor(5, 5)
xx = torch.autograd.Variable(x)
xy = torch.autograd.Variable(y)
print(type(x), type(xx))
torch.mm(xx, xy)

from time import time
import torch
import numpy as np

def test_sparse_speed(a1, a2, b2, density_A, density_B):
    b1 = a2
    nnz_A = int(density_A * a1 * a2)
    nnz_B = int(density_B * b1 * b2)
    
    print(f'Matrix A shape: [{a1}, {a2}], nonzero entries: {nnz_A}')
    print(f'Matrix B shape: [{b1}, {b2}], nonzero entries: {nnz_B}\n')
    
    # create first matrix
    indexA = np.array([np.random.randint(0, a1, (nnz_A, )), np.random.randint(0, a2, (nnz_A, ))])
    valueA = torch.ones(nnz_A, requires_grad=True)
    A = torch.sparse_coo_tensor(
        indexA,
        valueA,
        size=[a1, a2],
    )

    # create second matrix
    indexB = np.array([np.random.randint(0, b1, (nnz_B, )), np.random.randint(0, b2, (nnz_B, ))])
    valueB = torch.ones(nnz_B, requires_grad=True)
    B = torch.sparse_coo_tensor(
        indexB,
        valueB,
        size=[b1, b2],
    )
    
    start = time()
    y = torch.sparse.mm(A, B)
    print(f'Multiplication time: {time()-start} seconds')
    
    loss = y.coalesce().values().sum()
    
    start = time()
    loss.backward()
    print(f'Backward time: {time()-start} seconds')
    return A, B

A, B = test_sparse_speed(a1=100, a2=10**5, b2=10**4, density_A=0.005, density_B=0.0002)