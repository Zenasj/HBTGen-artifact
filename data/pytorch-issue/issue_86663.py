import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.LongTensor([1])
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])

embedding = nn.Embedding.from_pretrained(weight)  # this is ok 
out = embedding(input)

out = F.embedding(input, weight).squeeze(-1)  # this is ok   

weight = torch.LongTensor([[1, 2.3, 3], [4, 5.1, 6.3]])

embedding = nn.Embedding.from_pretrained(weight) # this will fail, 
# RuntimeError: Only Tensors of floating point and complex dtype can require gradients

out = embedding(input)

out = F.embedding(input, weight).squeeze(-1)  # this is ok