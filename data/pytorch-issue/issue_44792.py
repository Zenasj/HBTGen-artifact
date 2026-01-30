import torch.nn as nn

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda:0')
model = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=64,
            max_norm=1.0,
        ).to(device)
ix = torch.arange(2).long().to(device)
out = model(ix.repeat(2000))

for p in model.parameters():
    print((p ** 2).sum(dim=1, keepdim=True) ** 0.5)
print(out.sum())