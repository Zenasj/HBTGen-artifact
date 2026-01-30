import torch.nn as nn

import torch

emb = torch.nn.EmbeddingBag(10, 5)
input = torch.randint(0, 10, size=(0, 3))
print(emb(input))