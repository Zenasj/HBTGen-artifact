import torch

def is_sparse(matrix):
    return matrix.layout != torch.strided