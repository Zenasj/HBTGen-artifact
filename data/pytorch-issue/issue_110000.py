import torch

torch.cuda.current_device()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["WORLD_SIZE"] = "1"
import torch