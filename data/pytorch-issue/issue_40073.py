import torch

with torch.cuda.amp.autocast():
    _output = model(_input)