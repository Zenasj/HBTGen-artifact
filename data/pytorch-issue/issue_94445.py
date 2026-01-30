import torch.nn as nn

py
import torch
from torch.func import jacrev

torch.manual_seed(420)

embedding = torch.rand((2, 3, 4))

def func(embedding):
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    output = torch.nn.functional.embedding_bag(embedding, indices)
    return output

print(func(embedding))
# tensor([[0.5251, 0.3466, 0.4291],
#         [0.5251, 0.3466, 0.4291]])

jacrev(func)(embedding)
# RuntimeError: Function EmbeddingBagBackward0 returned an invalid gradient at index 0 - 
# got [2, 3] but expected shape compatible with [2, 3, 4]