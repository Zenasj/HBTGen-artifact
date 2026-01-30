import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from torch.utils.data import DataLoader, RandomSampler

# Hook class
class ActiveGradsHook:
    def __init__(self, name):
        self.name = name
        # self.__name__ = None # THIS WILL FIX YOUR CRASH

    def __call__(self, grad):
        return grad.clone()

def bad_hook(grad):
    raise RuntimeError("Bad hook")

def train_new_neurons(model):
    # Generate hooks for each layer

    for name, param in model.named_parameters():
        hook = ActiveGradsHook(name)
        param.register_hook(hook)

    # Train simply
    train(model)


def train(model):
    inputs = torch.rand(2, 3)
    action_target = torch.rand(2, 2)

    action_output = model(inputs)

    action_output.sum().backward()



if __name__ == "__main__":
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    train_new_neurons(model)