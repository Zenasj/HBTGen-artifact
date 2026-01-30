import torch
A = torch.tensor([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0],
                  [-7.0, 8.0, -9.0]])

eigenvalues, eigenvectors = torch.linalg.eigh(A)
print("Eigenvectors:", eigenvectors)

Eigenvectors: tensor([[ 0.4096,  0.4340, -0.8024],
        [-0.5426,  0.8230,  0.1681],
        [ 0.7333,  0.3665,  0.5726]])

import torch
A = torch.tensor([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0],
                  [-7.0, 8.0, -9.0]])
A = A.cuda()
eigenvalues, eigenvectors = torch.linalg.eigh(A)
print("Eigenvectors:", eigenvectors)

Eigenvectors: tensor([[-0.4096,  0.4340,  0.8024],
        [ 0.5426,  0.8230, -0.1681],
        [-0.7333,  0.3665, -0.5726]], device='cuda:0')