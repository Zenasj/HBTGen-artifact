import torch.nn as nn

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

    def forward(self, float_1, view_1):
        logits = float_1 / 64.0
        float_1 = None
        loss = torch.nn.functional.cross_entropy(logits, view_1, ignore_index=5)
        view_1 = None
        logsumexp = logits.logsumexp(dim=-1)
        logits = None
        return [loss, logsumexp]


mod = Repro()


def load_args(reader):
    buf0 = reader.storage("6916e7f050fea19610e426255220b0187cd89bce", 8388608, device=device(type="cuda", index=0))
    reader.tensor(buf0, (512, 4096), requires_grad=True)  # float_1
    buf1 = reader.storage(
        "b38053ad07d8a3c8e925c9fcbedfdf88ed396aeb", 4096, device=device(type="cuda", index=0), dtype_hint=torch.int64
    )
    reader.tensor(buf1, (512,), dtype=torch.int64, is_leaf=True)  # view_1


load_args._version = 0

if __name__ == "__main__":
    from torch._dynamo.repro.after_dynamo import run_repro

    run_repro(
        mod,
        load_args,
        accuracy=False,
        command="run",
        autocast=False,
        backend="inductor",
    )