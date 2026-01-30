import torch.nn as nn
import torchvision
import numpy as np

import torch
from torchvision.models import resnet18
from torch.autograd import Variable

import models.model

model = models.model.Encoder(image_size=(3, 128, 128), h_size=[64, 128, 256, 512], z_dim=1)
model = model.cuda()



# dummy inputs for the example
input = Variable(torch.randn(2,3,128,128).cuda(), requires_grad=True)
target = Variable(torch.zeros(2).cuda())

# as usual
output = model(input)
loss = torch.nn.functional.mse_loss(output, target)

grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
# torch.autograd.grad does not accumuate the gradients into the .grad attributes
# It instead returns the gradients as Variable tuples.

# now compute the 2-norm of the grad_params
grad_norm = 0
for grad in grad_params:
    grad_norm += grad.pow(2).sum()
grad_norm = grad_norm.sqrt()

# take the gradients wrt grad_norm. backward() will accumulate
# the gradients into the .grad attributes
grad_norm.backward()

def non_lin(nn_module, normalization="weight", activation="SELU", feat_size=None):
    """Non Lienar activation unit"""

    if normalization == "batch":
        assert feat_size is not None
        nn_module.append(nn.BatchNorm2d(feat_size))
    elif normalization == "weight":
        nn_module[-1] = nn.utils.weight_norm(nn_module[-1])
    elif normalization == "instance":
        assert feat_size is not None
        nn_module.append(nn.InstanceNorm2d(feat_size))

    if activation == "LReLU":
        nn_module.append(nn.LeakyReLU(0.2))
    elif activation == "ELU":
        nn_module.append(nn.ELU())
    elif activation == "ReLU":
        nn_module.append(nn.ReLU())
    elif activation == "SELU":
        nn_module.append(nn.SELU())
    else:
        warnings.warn("Will not use any non linear activation function", RuntimeWarning)


class Encoder(nn.Module):
    def __init__(self, image_size, z_dim=256, h_size=(64, 128, 256)):
        super(Encoder, self).__init__()

        n_channels = image_size[0]
        img_size_new = np.array([image_size[1], image_size[2]])

        if not isinstance(h_size, list) and not isinstance(h_size, tuple):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        elif len(h_size) < 3:
            raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        ### Start block
        start_block = []

        start_block.append(nn.Conv2d(n_channels, h_size_bot, kernel_size=4, stride=2, padding=1, bias=False))
        non_lin(start_block, activation="LReLU", normalization="batch", feat_size=h_size_bot)

        self.start = nn.Sequential(*start_block)
        img_size_new = img_size_new // 2

        ### Middle block (Done until we reach ? x 4 x 4)
        self.middle_blocks = []

        for h_size_top in h_size[1:]:
            middle_block = []
            middle_block.append(nn.Conv2d(h_size_bot, h_size_top, kernel_size=4, stride=2, padding=1, bias=False))
            non_lin(middle_block, activation="LReLU", normalization="batch", feat_size=h_size_top)

            middle = nn.Sequential(*middle_block)
            self.middle_blocks.append(middle)
            self.add_module("middle" + str(h_size_top), middle)

            h_size_bot = h_size_top
            img_size_new = img_size_new // 2

            if np.min(img_size_new) < 2:
                raise AttributeError("h_size to long, one image dimension has already perished")

        ### End block
        end_block = []

        end_block.append(
            nn.Conv2d(h_size_bot, z_dim, kernel_size=img_size_new.tolist(), stride=1, padding=0, bias=False))

        self.end = nn.Sequential(*end_block)

    def forward(self, inp):
        output = self.start(inp)
        for middle in self.middle_blocks:
            output = middle(output)
        output = self.end(output)
        return output