import torch

device = torch.device("mps")

###########################################
############# WORKING #####################
###########################################
n = 1024
# generate random positive semi-definite matrix
A = torch.rand(n, n, device=device)
A = torch.mm(A, A.t())
# invert
print("Before matrix inversion with n=1024")
V_pi = torch.linalg.inv(A)
print("After matrix inversion with n=1024")

###########################################
######### NOT WORKING #####################
###########################################
n = 1025
# generate random positive semi-definite matrix
A = torch.rand(n, n, device=device)
A = torch.mm(A, A.t())
# invert
print("Before matrix inversion with n=1025")
V_pi = torch.linalg.inv(A)                  # THIS FAILS
print("After matrix inversion with n=1025") # THIS IS NEVER REACHED