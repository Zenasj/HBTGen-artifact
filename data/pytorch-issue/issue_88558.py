#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:26:38 2022

@author: mabbbs
"""

# Imports

# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo
import random

# helper function to train a model
def train_model(model, trainloader):
    '''
    Function trains the model and prints out the training log.
    INPUT:
        model - initialized PyTorch model ready for training.
        trainloader - PyTorch dataloader for training data.
    '''
    #setup training

    #define loss function
    criterion = nn.NLLLoss()
    #define learning rate
    learning_rate = 0.003
    #define number of epochs
    epochs = 5
    #initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #run training and print out the loss to make sure that we are actually fitting to the training set
    print('Training the model. Make sure that loss decreases after each epoch.\n')
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            # print out the loss to make sure it is decreasing
            print(f"Training loss: {running_loss}")

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input) # simply apply already implemented SiLU

# simply define a silu function
def functional_silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise (functional)
    '''
    return F.silu(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

class Functional_SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return functional_silu(input) # simply apply already implemented SiLU

def main():
    
    print('Loading the Fasion MNIST dataset.\n')
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data for Fashion MNIST
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

    # 1. SiLU demonstration with model created with Sequential
    # use SiLU with model created with Sequential

    # initialize activation function
    activation_function = SiLU()
    #activation_function = Functional_SiLU()

    # Initialize the model using nn.Sequential
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(784, 256)),
                          ('activation1', activation_function), # use SiLU
                          ('fc2', nn.Linear(256, 128)),
                          ('bn2', nn.BatchNorm1d(num_features=128)),
                          ('activation2', activation_function), # use SiLU
                          ('dropout', nn.Dropout(0.3)),
                          ('fc3', nn.Linear(128, 64)),
                          ('bn3', nn.BatchNorm1d(num_features=64)),
                          ('activation3', activation_function), # use SiLU
                          ('logits', nn.Linear(64, 10)),
                          ('logsoftmax', nn.LogSoftmax(dim=1))]))

    # Run training
    print('Training model with SiLU activation.\n')
    train_model(model, trainloader)
    

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import platform

def silu1(x):
    m = nn.SiLU()
    return m(x)

def silu2(x):
    sigmoid = nn.Sigmoid()
    return x * sigmoid(x)

def main():
    torch.random.manual_seed(0)
    pc = platform.uname()
    pc_os = pc.system
    pc_version = pc.version
    pc_arch = pc.machine
    print(f'torch version:{torch.__version__}')
    print(f'OS:{pc_os}, version:{pc_version}, arch:{pc_arch}')
    
    x = torch.randn(2)
    silu1_result = silu1(x)
    silu2_result = silu2(x)
    print(f'diff(cpu) silu1 - silu2={silu1_result - silu2_result}')

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import platform

def silu1(x):
    m = nn.SiLU()
    return m(x)
def sigmoid(x):
    return 1 /  (1 + torch.exp(-x))
def silu2(x):
    # return x * sigmoid(x) # not match
    return x * 1 /  (1 + torch.exp(-x)) # match

def main():
    torch.random.manual_seed(0)
    pc = platform.uname()
    pc_os = pc.system
    pc_version = pc.version
    pc_arch = pc.machine
    print(f'torch version:{torch.__version__}')
    print(f'OS:{pc_os}, version:{pc_version}, arch:{pc_arch}')
    
    x = torch.randn(20)
    silu1_result = silu1(x)
    silu2_result = silu2(x)
    print(f'diff(cpu) silu1 - silu2={silu1_result - silu2_result}')

if __name__ == '__main__':
    main()