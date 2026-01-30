import torchvision

import torch
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter


model = resnet50()
writer = SummaryWriter("/home/tensorboard_test")
board_input = torch.rand((3, 224, 224)).unsqueeze(0)
writer.add_graph(model, board_input)
writer.close()