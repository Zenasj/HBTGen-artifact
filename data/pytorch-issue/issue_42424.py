import torch.nn as nn
d_model = 4
nhead = 2
dropout = 0.0
net = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
for m in net.modules():
    print('Module name :', m)
    for nm, p in m.named_parameters(recurse=False):
        print(nm, ":", p)