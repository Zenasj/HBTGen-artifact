import torch
import random

import numpy as np 
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter( )


for i in range(10):
    writer.add_scalar('loss/test',np.random.random() , i)