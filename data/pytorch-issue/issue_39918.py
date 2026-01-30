import os
import torch
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
print(torch.cuda.device_count())