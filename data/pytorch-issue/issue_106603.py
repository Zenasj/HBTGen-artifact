import numpy as np

import torch
# file
covariance_matrices = torch.load("covariance_matrices.pt")
c = covariance_matrices[43,1,-8]
print("-"*50)
print("c: \n", c)
print("-"*50)
print("det: \n",torch.det(c))
print("-"*50)
print("index det: \n",c[0,0]*c[1,1] - c[0,1]*c[1,0])
print("-"*50)
print("compute det manualy: \n",(1.7533e+03)*(9.9205e+17) - (4.1705e+10)*(4.1705e+10))

print("-"*50)
n = c.detach().cpu().numpy()
print("numpy det: \n",np.linalg.det(n))
print(n)

print("-"*50)
t = torch.tensor([[1.7533e+03, 4.1705e+10],
        [4.1705e+10, 9.9205e+17]])

print("new tensor det: \n", torch.det(t))