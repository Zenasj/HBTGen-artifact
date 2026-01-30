import torch.nn as nn

import torch
from torch import sparse
from torch.nn import init
from torch import nn
from itertools import count


def main():
    idx = torch.LongTensor([[], []])
    values = torch.FloatTensor([])
    sp_tensor = sparse.FloatTensor(idx, values, torch.Size([3, 10]))
    mat2 = nn.Parameter(init.xavier_normal_(torch.empty(10, 150)))
    print(sp_tensor.to_dense())
    for i in count():
        spmm = sparse.mm(sp_tensor, mat2)
        assert bool(torch.sum(torch.isnan(spmm)) == 0), "iter: {} ===> sparse mm nan happened.".format(i)


if __name__ == '__main__':
    main()