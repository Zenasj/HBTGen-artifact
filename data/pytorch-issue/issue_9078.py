import torch.nn as nn
import numpy as np

import torch
torch.manual_seed(6)

from onmt.modules.embeddings import Embeddings

class Test(torch.nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        self.dense = torch.nn.Linear(100, 1)
        self.oembd = Embeddings(word_vec_size=100,
                   position_encoding=False,
                   dropout=0.3,feat_merge=None,
                   word_padding_idx=1,
                   word_vocab_size=1000)

    def forward(self, inp):
        inp = self.oembd(inp)
        
        return self.dense(inp[0])

test = Test()
test.cuda()
inp = torch.tensor([1, 1, 2, 1, 1, 2],device='cuda')
inp=inp.unsqueeze(0).unsqueeze(-1)
out = test(inp)
raw_loss = out.mean(dim=1)
loss_grad = torch.autograd.grad(outputs=raw_loss,
                                inputs=list(test.parameters()),
                                retain_graph=True, create_graph=True, only_inputs=True)

norm = sum([param.norm()**2 for param in loss_grad])

loss = raw_loss + norm

loss.backward()
print("Succesful")