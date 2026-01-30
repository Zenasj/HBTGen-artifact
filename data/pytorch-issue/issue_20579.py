import torch

from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as writer:
    writer.add_custom_scalars({'Stuff': {
        'Losses': ['MultiLine', ['loss/(one|two)']],
        'Metrics': ['MultiLine', ['metric/(three|four)']],
    }})