import torch

torch.manual_seed(42)

sumtokens = torch.randn(512, 32)
tokenids = torch.arange(0,472)
tindex = torch.arange(0,472)
tokens = torch.randn(512,32)

sumtokens[tokenids] += tokens[tindex]

print(sum(map(sum,sumtokens)))