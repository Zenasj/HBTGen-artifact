import torch.nn as nn

import torch.nn.functional as F
import torch
from torch import nn

class GumbelVectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.num_vars = 320

        self.weight_proj = nn.Linear(256, self.num_groups * self.num_vars)
        self.temperature = 2

        self.weight_proj.weight.data.normal_(mean=0.0, std=1)
        self.weight_proj.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        codevector_probs = F.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)
        return codevector_probs


vq = GumbelVectorQuantizer().cuda()
vq_compiled = torch.compile(vq)

for i in range(1000):
    x = torch.randn(4, 400, 256).cuda()
    out = vq(x)
    out_compiled = vq_compiled(x)

    if out.isnan().any() or out_compiled.isnan().any():
        print(f"{i} - eager: {out.isnan().sum()} NaNs, compiled: {out_compiled.isnan().sum()} NaNs")
        break

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().add_(eps).log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


F.gumbel_softmax = gumbel_softmax

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    with torch._dynamo.utils.preserve_rng_state():
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret