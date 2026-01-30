import torch.nn as nn

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel, self).__init__()
        dim = 5
        self.emb = nn.Embedding(10, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
        return self.seq(input)

model = SomeModel()

dummy_input = torch.tensor([2], dtype=torch.long)
dummy_input

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel, self).__init__()
        dim = 5
        self.seq = nn.Sequential(
            nn.Embedding(10, dim),
            nn.Linear(dim, 1),
        )

    def forward(self, input):
        return self.seq(input)

model = SomeModel()

import os

import torch
from torch import nn


class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel, self).__init__()
        dim = 5
        self.emb = nn.Embedding(10, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
        return self.seq(input)


model = SomeModel()


dummy_input = torch.tensor([2], dtype=torch.long)
dummy_input

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
os.remove('model.onnx')


print('\n\nTEST CASE 2\n\n')


n = 8
dim = 10

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel, self).__init__()
        self.embedding = nn.Embedding(n, dim)
        self.seq = nn.Sequential(
            self.embedding,
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, indices):
        return self.seq(indices)

model = SomeModel()

dummy_input = torch.LongTensor([2])
torch.onnx.export(model, dummy_input, "foo.onnx", verbose=True)

os.remove('foo.onnx')