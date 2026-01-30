import torch.nn as nn

def accuracy(out, labels):
    assert out.ndim == 2
    assert out.size(0) == labels.size(0)
    assert labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1)
    labels = labels.flatten()
    predictions = torch.argmax(out, 1)
    return (labels == predictions).sum(dtype=torch.float64) / labels.size(0)

@torch.compile
def train_step(minibatch, optimizer, model, loss_fn):
    category = "paper"
    node_features = {
        ntype: feat.float()
        for (ntype, name), feat in minibatch.node_features.items()
        if name == "feat"
    }
    labels = minibatch.labels[category].long()
    optimizer.zero_grad()
    out = model(minibatch.sampled_subgraphs, node_features)[category]
    loss = loss_fn(out, labels)
    # https://github.com/pytorch/pytorch/issues/133942
    # num_correct = accuracy(out, labels) * labels.size(0)
    num_correct = torch.zeros(1, dtype=torch.float64, device=out.device)
    loss.backward()
    optimizer.step()
    return loss.detach(), num_correct, labels.size(0)

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config









from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L__self___L__model___layers_2_dropout = Dropout(p=0.5, inplace=False)
        self.L__self___loss_fn = CrossEntropyLoss()



    def forward(self, L_L_labels_ : torch.Tensor, x_7, labels):
        l_l_labels_ = L_L_labels_
        out_30 = self.L__self___L__model___layers_2_dropout(x_7);  x_7 = None
        loss = self.L__self___loss_fn(out_30, l_l_labels_);  l_l_labels_ = None
        predictions = torch.argmax(out_30, 1);  out_30 = None
        eq = labels == predictions;  labels = predictions = None
        sum_1 = eq.sum(dtype = torch.float64);  eq = None
        truediv_15 = sum_1 / 1024;  sum_1 = None
        num_correct = truediv_15 * 1024;  truediv_15 = None
        return [loss, num_correct]


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('a1b9334d88e7a05117e3540e1976a55e42a3f277', 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1024,), dtype=torch.int64, is_leaf=True)  # L_L_labels_
    buf1 = reader.storage('3a396a0bd2d1e5fc6bb833c38aef3e0dd147a109', 626688, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1024, 153), requires_grad=True)  # x_7
    reader.tensor(buf0, (1024,), dtype=torch.int64, is_leaf=True)  # labels
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/localscratch/dgl-3/examples/graphbolt/pyg/labor/checkpoints', autocast=False, backend='inductor')