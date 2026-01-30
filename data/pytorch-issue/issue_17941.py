import torch
import torch.nn as nn

m = nn.Embedding(10, 3, sparse=True)
m.zero_grad()
m(torch.LongTensor([7, 1, 3])).sum().backward()
print(m.weight.grad)

m = nn.Embedding(10, 3, sparse=True)
m.zero_grad()
m(torch.LongTensor([7, 1, 3])).sum().backward()
m(torch.LongTensor([7, 1, 3])).sum().backward()
print(m.weight.grad)

m = nn.Embedding(10, 3, sparse=True)
m.zero_grad()
m(torch.LongTensor([7, 1, 3])).sum().backward()
m(torch.LongTensor([8, 1, 3])).sum().backward()
print(m.weight.grad)