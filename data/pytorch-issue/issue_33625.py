import random

import torch.nn as nn
from numpy import prod

dim = 29
layers = 5
use_bias = True

bigru = nn.GRU(dim, dim, layers, bidirectional=True, bias=use_bias)
bigru_params = sum(prod(p.size()) for p in bigru.parameters())

bigru_docs = 2*layers*(dim*dim + dim*dim +    # Wir, Whr in docs
                       use_bias*(dim + dim) + # bir, bhr in docs
                       dim*dim + dim*dim +    # Wiz, Whz in docs
                       use_bias*(dim + dim) + # biz, bhz in docs
                       dim*dim + dim*dim +    # Win, Whn in docs
                       use_bias*(dim + dim))  # bin, bnn in docs
                       
gru = nn.GRU(dim, dim, layers, bidirectional=False, bias=use_bias)
gru_stacked_params = 2 * sum(prod(p.size()) for p in gru.parameters())

print('\n'.join([
    'BiGRU Params: {:d}'.format(bigru_params),
    'Expected BiGRU Params: {:d}'.format(bigru_docs),
    'Stacked GRU Params: {:d}'.format(gru_stacked_params)]))

import torch
import torch.nn as nn
from numpy import prod

torch.random.manual_seed(2718)

dim = 29
layers = 3
use_bias = True
bigru = nn.GRU(dim, dim, layers, bidirectional=True, bias=use_bias)

seq_len = 5
batch_size = 7
ntokens = 11

emb = nn.Embedding(ntokens, dim)

tokens = torch.randint(low=0, high=ntokens-1, size=(seq_len, batch_size))
tokens_emb = emb(tokens)

out_prefill, _ = bigru(tokens_emb)
out_prefill = out_prefill.view(seq_len, batch_size, 2, dim)

bigru.weight_ih_l0_reverse.data.fill_(100.)
bigru.weight_hh_l0_reverse.data.fill_(100.)
bigru.weight_ih_l1_reverse.data.fill_(100.)
bigru.weight_hh_l1_reverse.data.fill_(100.)

out_postfill, _ = bigru(tokens_emb)
out_postfill = out_postfill.view(seq_len, batch_size, 2, dim)

lr_diff = (out_prefill[:, :, 0, :] -
           out_postfill[:, :, 0, :])

rl_diff = (out_prefill[:, :, 1, :] -
           out_postfill[:, :, 1, :])

print('Forward Mean Abs Diff: {:.3f}'.format(lr_diff.mean().item()))
print('Backward Mean Abs Diff: {:.3f}'.format(rl_diff.mean().item()))