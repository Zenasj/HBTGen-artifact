import torch.nn as nn

import torch
dev = torch.device('cuda')
torch.manual_seed(0)

random_ix = torch.randint(high=10, size=(256, 3, 7))
embedding_layer = torch.nn.Embedding(10, 5, padding_idx=0)

embedding_layer.to(dev)
random_ix = random_ix.to(dev)

embeds = embedding_layer(random_ix)
merged = torch.sum(embeds, dim=2)
summed = merged.sum()
summed.backward()

print(embedding_layer.weight.grad[0])

torch.manual_seed(0)

random_ix = torch.randint(high=10, size=(256, 3, 7))
embedding_layer = torch.nn.Embedding(10, 5, padding_idx=0)

embeds = embedding_layer(random_ix)
merged = torch.sum(embeds, dim=2)
summed = merged.sum()
summed.backward()

print(embedding_layer.weight.grad[0])