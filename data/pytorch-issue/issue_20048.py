from torch.multiprocessing import Process
model = loadModel() # somehow load a model (e.g. from torch vision)
inputList = loadListOfInputs() # somehow get the list of input tensors

processes = []
for i in range(100):
    processes.append(Process(target=doForwardPass, kwargs={'input': inputList[i]}))
    processes[-1].start()

for i in range(100):
    processes[i].join()

def doForwardPass(input):
    # model.cuda() # this, uncommented, used to work also
    output = model(input)

import torch
import os

print(torch.zeros(2))

os.fork()

print(torch.zeros(2, device='cuda'))

import torch
from torch.multiprocessing import Process
import torchvision

model = torchvision.models.resnet50()
inputList = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]

def doForwardPass(input):
    model.cuda()

processes = []
for i in range(2):
    processes.append(Process(target=doForwardPass, kwargs={'input': inputList[i]}))
    processes[-1].start()

for i in range(2):
    processes[i].join()