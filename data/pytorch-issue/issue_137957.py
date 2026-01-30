import torchvision

import torch
  
from torch.cuda._memory_viz import compare
from torchvision.models import resnet18

def do_snapshot(bs):
    torch.cuda.memory._record_memory_history()
    model = resnet18().cuda()
    input = torch.rand(bs, 3, 224, 224).cuda()
    output = model(input)
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.empty_cache()
    return snapshot

sn1 = do_snapshot(1)
sn2 = do_snapshot(2)
compare(sn1, sn2)