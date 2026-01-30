import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x, y):
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        return loss_fct(x, y)

model = Model()
model = torch.compile(model)
y = model(torch.rand(3, 10), torch.randint(high=10, size=(3,)))