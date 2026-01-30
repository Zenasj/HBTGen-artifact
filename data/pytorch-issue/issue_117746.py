import os
import torch
from torch.distributed._tensor.device_mesh import DeviceMesh
os.environ['RANK'] = '0'; os.environ['WORLD_SIZE'] = '1'; os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '25364'
DeviceMesh("cpu", torch.arange(1))