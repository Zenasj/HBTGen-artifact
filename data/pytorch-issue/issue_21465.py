import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer.add_embedding(torch.randn(100, 5))
writer.close()