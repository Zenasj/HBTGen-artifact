import torch
import torch.nn as nn

b = nn.Embedding(10, 20, 0)
b.half()
b(torch.randint(0, 10, size=(5, 10)))