import torch

torch.linalg.svd(torch.zeros(3,3).cuda(), full_matrices=False, driver='gesvda')