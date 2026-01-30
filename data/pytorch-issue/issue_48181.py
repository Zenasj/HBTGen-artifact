import torch.nn as nn

import torch

linear = torch.nn.Linear(3, 4)
norm_layer = torch.nn.utils.spectral_norm(linear)

print('Normalized weight matrix with spectral_norm(): ', norm_layer.weight)
print('Original weight matrix: ', norm_later.weight_orig)

sigma = torch.dot(norm_layer.weight_u, torch.mv(norm_layer.weight_orig, norm_layer.weight_v))
print('Normalized weight matrix by hands: ', a.weight_orig / sigma)