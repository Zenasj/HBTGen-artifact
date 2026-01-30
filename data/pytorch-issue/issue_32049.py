import torch.nn as nn

import torch
import torch.nn.functional as F

logits = torch.randn(5, 2).log_softmax(-1)

F.nll_loss(logits, torch.tensor([0, 1, 0, 1, 0]), reduction='none') # works as intended
F.nll_loss(logits, torch.tensor([0, 1, 0, 1, -100]), reduction='none') # works as intended
F.nll_loss(logits, torch.tensor([0, 1, 0, 1, 100]), reduction='none') # raises exception as intended
F.nll_loss(logits, torch.tensor([0, 1, 0, 1, -99]), reduction='none') # should but does not raise an exception, returns garbage