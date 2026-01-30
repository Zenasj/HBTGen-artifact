import torch.nn as nn
import torchvision

import torch, torchvision

# The following instantiates activation_layer with inplace=True, by default
torchvision.ops.misc.Conv2dNormActivation(1, 1, activation_layer=torch.nn.ReLU)  # works
torchvision.ops.misc.Conv2dNormActivation(1, 1, activation_layer=torch.nn.GELU)  # fails with TypeError