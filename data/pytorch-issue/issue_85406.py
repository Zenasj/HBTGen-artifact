import torch

N = 3

A = torch.rand([N,N],dtype=torch.float32)*2 - 1
B = torch.rand([N,N],dtype=torch.float32)*5 - 2.5

C = A @ B


device = torch.device('mps')

aa = A.to(device)
bb = B.to(device)
cc = aa @ bb

# this is aa on GPU
tensor([[ 0.5255, -0.9316,  0.0885],
        [ 0.9530, -0.7145, -0.7237],
        [ 0.3007,  0.9739, -0.9347]], device='mps:0')

# this is A on CPU
tensor([[ 0.5255, -0.9316,  0.0885],
        [ 0.9530, -0.7145, -0.7237],
        [ 0.3007,  0.9739, -0.9347]])

# cc -- the multiplication on GPU

tensor([[ 0.6661, -0.5107, -0.8595],
        [ 0.1683,  1.4892,  0.1642],
        [-0.9340,  2.8247,  1.6085]], device='mps:0')

# C -- the multiplication on CPU

tensor([[-0.8219, -1.0997,  1.7780],
        [ 0.1301, -1.3871,  2.9537],
        [ 1.7954,  0.4726,  0.0359]])

mat_A = torch.Tensor([[1,2,3],
                      [4,5,6],
                      [7,8,9]])

mat_B = torch.Tensor([[9,8,7],
                      [6,5,4],
                      [5,4,3]])

mat_C = mat_A @ mat_B

# now let's put the matrice mat_A and mat_B to GPU:

mps_A = mat_A.to(device)
mps_B = mat_B.to(device)
mps_C = mps_A @ mps_B

print(mps_A)
print(mps_B)

tensor([[1.0000, 2.0000, 3.0000],
        [4.0000, 5.0000, 6.0000],
        [7.0000, 8.0000, 9.0000]], device='mps:0')
tensor([[9.0000, 8.0000, 7.0000],
        [6.0000, 5.0000, 4.0000],
        [5.0000, 4.0000, 3.0000]], device='mps:0')

print('mat_C on cpu:',mat_C)
print('mps_C on gpu:',mps_C)

tensor([[ 46.,  28.,  22.],
        [118.,  73.,  58.],
        [190., 118.,  94.]])

import torch

N = 3

A = torch.rand([N, N], dtype=torch.float32)*2 - 1
B = torch.rand([N, N], dtype=torch.float32)*5 - 2.5

C = A @ B.T


device = torch.device('mps')

aa = A.to(device)
bb = B.to(device)
cc = aa @ bb

print(cc.cpu() - C)

tensor([[ 1.1921e-07,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00, -1.1921e-07,  2.7940e-09]])

import torch

N = 3

A = torch.rand([N, N], dtype=torch.float32)*2 - 1
B = torch.rand([N, N], dtype=torch.float32)*5 - 2.5

# C = A @ B.T

C = torch.mm(A, B.T)


device = torch.device('mps')

aa = A.to(device)
bb = B.to(device)
# cc = aa @ bb

cc = torch.mm(aa, bb)

print(cc.cpu()-C)
# results:(also only truncated errors)

tensor([[ 1.1921e-07,  0.0000e+00, -8.5682e-08],
        [-5.9605e-08,  0.0000e+00,  1.1921e-07],
        [ 0.0000e+00,  0.0000e+00,  1.1921e-07]])

import torch

N = 3

A = torch.rand([N, N], dtype=torch.float32)*2 - 1
B = torch.rand([N, N], dtype=torch.float32)*5 - 2.5

# C = A @ B.T

C = torch.mm(A, B)


device = torch.device('mps')

aa = A.to(device)
bb = B.to(device)
# cc = aa @ bb

cc = torch.mm(aa, bb)

print(cc.cpu()-C)
print('mse:', ((cc.cpu()-C)**2).sum())

# results:(only truncated errors appear again)
tensor([[-1.4901e-08,  2.2352e-08,  0.0000e+00],
        [ 0.0000e+00, -2.3842e-07,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])
mse: tensor(5.7565e-14)