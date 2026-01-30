py
import torch

A = torch.tensor([[0.0100, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0100, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0100, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0100, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0100]])

for i in range(1000):
    print(i)
    (eigvals, eigvecs) = torch.lobpcg(A)

print(eigvals), print(eigvecs)

py
import torch

A = torch.tensor([[0.0100, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0100, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0100, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0100, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0100]]).cuda()

for i in range(1000):
    print(i)
    (eigvals, eigvecs) = torch.lobpcg(A)

print(eigvals), print(eigvecs)

tensor([0.0100], device='cuda:0')
tensor([[-0.3277],
        [-0.0372],
        [ 0.1163],
        [-0.4553],
        [-0.8188]], device='cuda:0')