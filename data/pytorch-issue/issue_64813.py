import torch
import torch.nn as nn

class Example:
    def __init__(self):
        self.x = 0
        
x = torch.tensor([1., 2., 3., 4., 5.])
model = torch.nn.Linear(5, 1)
model(x).backward()

model.weight.grad = Example()