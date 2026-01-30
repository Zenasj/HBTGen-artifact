import torch.nn as nn

from functorch.experimental import functionalize
import torch

def f(x):
    log_probs = x.log_softmax(2)
    targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
    input_lengths = torch.full((16,), 50, dtype=torch.long)
    target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return loss

f_functional = functionalize(f, remove="mutations_and_views")
f_functional(torch.randn(50, 16, 20))