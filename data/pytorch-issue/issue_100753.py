import torch.nn as nn

py
import math
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self, seq_len, d_model):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.attn_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(-1, self.seq_len, self.d_model)
        key = key.view(-1, self.seq_len, self.d_model)
        value = value.view(-1, self.seq_len, self.d_model)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_model)
        scores.masked_fill_(~self.attn_mask[None, None, :, :], float('-inf'))
        return scores
seq_len = 8
d_model = 16
batch_size = 1

x = torch.randn((batch_size, seq_len, d_model))
mask = torch.randint(0, 2, (1, batch_size, seq_len, d_model)).bool()
func = Model(seq_len, d_model).to('cpu')

with torch.no_grad():
    func.train(False)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # success

    res1 = func(x) # without jit
    print(res1)
    # RuntimeError: output with shape [1, 8, 8] doesn't match the broadcast shape [1, 1, 8, 8]