import torch
from torch import nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=True)
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/deeplabv3')
x = torch.rand(1,3,512,512)
writer.add_graph(deeplabv3, x)