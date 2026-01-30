import torch.nn as nn

import torch

a_torch = torch.tensor([2606.66824394, 2477.72226966, 3251.84008903], dtype=torch.float64)
b_torch = torch.tensor([3.46822161e-09, 1.82536693e-09, 8.35245752e-09], dtype=torch.float64)
cos_sim_torch = torch.nn.functional.cosine_similarity(a_torch, b_torch, dim=0)
print(f"PyTorch Cosine Similarity: {cos_sim_torch.item()}")

import numpy as np
print(np.__version__)
a_np = np.array([2606.66824394, 2477.72226966, 3251.84008903])
b_np = np.array([3.46822161e-09, 1.82536693e-09, 8.35245752e-09])
norm_a = np.linalg.norm(a_np)
if norm_a < 1e-8:
    norm_a = 1e-8
norm_b = np.linalg.norm(b_np)
if norm_b < 1e-8:
    norm_b = 1e-8
print(norm_a, norm_b)
cos_sim_np = np.dot(a_np, b_np) / (norm_a * norm_b)
print(f"NumPy Cosine Similarity: {cos_sim_np}")