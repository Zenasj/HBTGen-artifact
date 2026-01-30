import torch
import torch.nn as nn

@torch.compile
def function(X):
    embedding = torch.nn.Embedding(7, 9).cuda()
    return embedding(X)

function(torch.tensor([10]).cuda())