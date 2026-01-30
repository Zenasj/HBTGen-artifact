import torch
import torch.nn as nn

# make a chain of BN to show increase the differences
bnlist2 = nn.Sequential(*[nn.BatchNorm1d(2, eps=0, momentum=0.01, affine=False) for _ in range(20)])
bnlist2.train()

# setup a random x
x = torch.randn(64, 2) * 2 - 1

# running 2048 times to make sure the `running_means` and `running_vars` be stable.
for _ in range(2048):
    y = bnlist2(x)


# calculate the output at different mode
bnlist2.train()
y = bnlist2(x)

bnlist2.eval()
y_eval = bnlist2(x)

mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)

((x - mean) / (var + 0).sqrt())[:5]