import torch.nn as nn

import torch
ce = torch.nn.CrossEntropyLoss().cuda()
logits = torch.randn(int(2**16), int(2**16)+1, dtype=torch.float32, device='cuda')
labels = torch.zeros(logits.size(0), dtype=torch.long, device='cuda')
ce(logits, labels)