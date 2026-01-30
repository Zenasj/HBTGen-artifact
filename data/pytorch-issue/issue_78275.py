import torch

device = torch.device('cuda')
z = torch.tensor([0.0, 3.14]) 
S = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

print(torch.einsum('i,ij,j->',z,S,z))
print(z.t() @ S @ z)
print(z[0]*S[0,0]*z[0]+z[0]*S[0,1]*z[1]+z[1]*S[1,0]*z[0]+z[1]*S[1,1]*z[1])

z = z.to(device)
S = S.to(device)

print(torch.einsum('i,ij,j->',z,S,z))
print(z.t() @ S @ z)
print(z[0]*S[0,0]*z[0]+z[0]*S[0,1]*z[1]+z[1]*S[1,0]*z[0]+z[1]*S[1,1]*z[1])

tensor(9.8596)
tensor(9.8596)
tensor(9.8596)
tensor(9.8596, device='cuda:0')
tensor(9.8596, device='cuda:0')
tensor(9.8596, device='cuda:0')

tensor(9.8596)
tensor(9.8596)
tensor(9.8596)
tensor(9.8616, device='cuda:0')
tensor(9.8616, device='cuda:0')
tensor(9.8596, device='cuda:0')