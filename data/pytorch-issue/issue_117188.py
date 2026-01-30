import torch.nn as nn

import torch
import torch.export
import torch.sparse


class BikNet(torch.nn.Module):

  def __init__(self):
    super(BikNet, self).__init__()
    return

  def forward(self, x):
    return x.sum()


biknet = BikNet()
biknet.eval()

dense_input = torch.ones(64, 64)
sparse_input = dense_input.to_sparse_csr()

prog1 = torch.export.export(biknet, args=(dense_input,))
prog2 = torch.export.export(biknet, args=(sparse_input,))   # fails