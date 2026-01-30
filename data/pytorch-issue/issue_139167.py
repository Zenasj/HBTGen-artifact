import torch

torch._dynamo.config.inline_inbuilt_nn_modules = False