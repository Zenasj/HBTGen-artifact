import torch.nn as nn

import torch
print (torch.__version__)

_ = torch.manual_seed (2021)

nBatch = 128
nClass = 2
height = 768
width = 768

input = torch.randn (nBatch, nClass, height, width)
target = torch.randint (nClass, (nBatch, height, width))

print ('nBatch =', nBatch, '\nnClass =', nClass, '\nheight =', height, '\nwidth =', width)
print ('big, float:')
print ('mean:', torch.nn.CrossEntropyLoss() (input, target))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input, target).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input, target) / target.numel())

print ('big, cuda:')
print ('mean:', torch.nn.CrossEntropyLoss() (input.cuda(), target.cuda()))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input.cuda(), target.cuda()).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input.cuda(), target.cuda()) / target.numel())

print ('big, double:')
print ('mean:', torch.nn.CrossEntropyLoss() (input.double(), target))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input.double(), target).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input.double(), target) / target.numel())


nBatch = 8
nClass = 2
height = 16
width = 16

input = torch.randn (nBatch, nClass, height, width)
target = torch.randint (nClass, (nBatch, height, width))

print ('nBatch =', nBatch, '\nnClass =', nClass, '\nheight =', height, '\nwidth =', width)
print ('small, float:')
print ('mean:', torch.nn.CrossEntropyLoss() (input, target))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input, target).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input, target) / target.numel())

print ('small, cuda:')
print ('mean:', torch.nn.CrossEntropyLoss() (input.cuda(), target.cuda()))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input.cuda(), target.cuda()).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input.cuda(), target.cuda()) / target.numel())

print ('small, double:')
print ('mean:', torch.nn.CrossEntropyLoss() (input.double(), target))
print ('none:', torch.nn.CrossEntropyLoss (reduction = 'none') (input.double(), target).mean())
print ('sum:', torch.nn.CrossEntropyLoss (reduction = 'sum') (input.double(), target) / target.numel())