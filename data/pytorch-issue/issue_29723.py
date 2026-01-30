from copy import deepcopy
import numpy as np
import torch


def init_matrices(n: int = 10000, k: int = 300):
    torch.manual_seed(1)
    S = torch.rand(n, n)
    A = torch.rand(n, k)
    R = torch.rand(k, k)
    return S, A, R


S, A, R = init_matrices()
S_, A_, R_ = np.array(deepcopy(S)), np.array(deepcopy(A)), np.array(deepcopy(R))

N = A_.dot(R_).dot(A_.T)
T = A.matmul(R).matmul(A.transpose(0, 1))

print('np.linalg.norm(N)', np.linalg.norm(N))                           # correct
print('torch.norm(T)', torch.norm(T))                                   # wrong
print('torch.sqrt(torch.sum(T ** 2))', torch.sqrt(torch.sum(T ** 2)))   # correct

print('np.linalg.norm(A_)', np.linalg.norm(A_))                         # correct
print('np.linalg.norm(A_.T)', np.linalg.norm(A_.T))                     # correct
print('np.linalg.norm(R_)', np.linalg.norm(R_))                         # correct

print('torch.norm(A)', torch.norm(A))                                   # correct
print('torch.norm(A.transpose(0, 1))', torch.norm(A.transpose(0, 1)))   # correct
print('torch.norm(R)', torch.norm(R))

from copy import deepcopy
import numpy as np
import torch


def init_matrices(n: int = 10000, k: int = 300):
    torch.manual_seed(1)
    S = torch.rand(n, n)
    A = torch.rand(n, k)
    R = torch.rand(k, k)
    return S, A, R


S, A, R = init_matrices()
S_, A_, R_ = np.array(deepcopy(S)), np.array(deepcopy(A)), np.array(deepcopy(R))

N = A_.dot(R_).dot(A_.T)
T = A.matmul(R).matmul(A.transpose(0, 1))

print('np.linalg.norm(N)', np.linalg.norm(N))                           # correct
print('torch.norm(T)', torch.norm(T))                                   # wrong
print('torch.sqrt(torch.sum(T ** 2))', torch.sqrt(torch.sum(T ** 2)))   # correct

print('np.linalg.norm(A_)', np.linalg.norm(A_))                         # correct
print('np.linalg.norm(A_.T)', np.linalg.norm(A_.T))                     # correct
print('np.linalg.norm(R_)', np.linalg.norm(R_))                         # correct

print('torch.norm(A)', torch.norm(A))                                   # correct
print('torch.norm(A.transpose(0, 1))', torch.norm(A.transpose(0, 1)))   # correct
print('torch.norm(R)', torch.norm(R))

T_ = np.array(deepcopy(T))

print('torch.norm(T_)', torch.norm(T))
print('torch.norm(deepcopy(T))', torch.norm(deepcopy(T)))

num = np.linalg.norm(
                    (extracted_feat[str(key)].cpu().detach().numpy() - extracted_feats_aug[str(key)].cpu().detach().numpy()) / np.mean(extracted_feat[str(key)].cpu().detach().numpy()))