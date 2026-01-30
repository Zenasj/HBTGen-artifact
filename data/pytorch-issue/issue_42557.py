import sys
import psutil
import torch

N = 5000
device = torch.device('cuda:0')
print('torch.symeig system memory:', end=' ')
for i in range(10):
    N += 1
    A = torch.eye(N).to(device)
    d, V = torch.symeig(A, eigenvectors=True)
    print('%.1f%%'%(psutil.virtual_memory().percent), end=' ')
    sys.stdout.flush()
print()

print('torch.eig    system memory:', end=' ')
for i in range(10):
    N += 1
    A = torch.eye(N).to(device)
    d, V = torch.eig(A, eigenvectors=True)
    print('%.1f%%'%(psutil.virtual_memory().percent), end=' ')
    sys.stdout.flush()
print()