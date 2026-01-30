import torch

t1 = tensor([[1.,2,3],[-4.,.5,6]])
t2 = tensor([[1.,-2],[3.,4],[5.,-6]])
A = torch.matmul(t2,t1)
# det is zero or near zero: A is singular
A.det()

b = torch.normal(mean=0.0, std=1.0, size=(3,1))

# solve Ax = b
LU_A = torch.lu(A)
x = torch.lu_solve(b, *LU_A) # this should give an error message that A is singular instead of spitting out garbage

# Verification: should return a vector of zeros, but doesn't
torch.matmul(A, x) - b