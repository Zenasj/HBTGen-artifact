import torch

def multiple_matrix_mult_elementwise(A, B, C):
    D = A
    for _ in range(5):
        D = torch.matmul(D, B)
    E = D + C
    for _ in range(5):
        E = E * torch.sigmoid(E)
    return E