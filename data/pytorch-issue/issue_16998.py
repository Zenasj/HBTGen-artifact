import torch.nn as nn

python
import torch
from torch import nn


def main():
    print('torch version', torch.version.__version__)
    print('cuda version', torch.version.cuda)

    bn = nn.BatchNorm1d(1)
    bn.cuda()
    for i in range(16, 20):
        N = 2 ** i
        print('i, N', i, N)
        A = torch.cuda.FloatTensor(N, 1)
        result = bn(A)


if __name__ == '__main__':
    main()