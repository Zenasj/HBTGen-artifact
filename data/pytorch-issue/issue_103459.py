import torch.nn as nn

import torch
from torch.nn.functional import gumbel_softmax


def test_gumbel():
    print(torch.__version__)
    torch.manual_seed(1234)
    randy = torch.rand((5,8192,96,96)) *2 -1
    randy *= 255

    print(randy.dtype, randy.shape, torch.min(randy), torch.max(randy))
    out = gumbel_softmax(randy)
    print(out.dtype, out.shape, torch.min(out), torch.max(out))

if __name__ == "__main__":
    test_gumbel()

# gumbels = (
#     -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
# )  # ~Gumbel(0,1)
gumbels = torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_() + eps
gumbels = (-gumbels.log())