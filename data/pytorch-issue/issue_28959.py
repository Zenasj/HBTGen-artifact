import random

import torch
import torch.nn as nn
import numpy as np


class sampleModel(nn.Module):
    def __init__(self):
        super(sampleModel, self).__init__()
        input_layer= nn.Embedding(500, 10, max_norm=1.0)
        output_layer = nn.Linear(10, 2)
        self.layers = nn.Sequential(input_layer, output_layer)
        self.loss_criterion = nn.NLLLoss()

    def forward(self, input, labels):
        em_x = self.layers[0](input).sum(1)
        out = self.layers[1](em_x)
        loss = self.loss_criterion(out, labels)
        return loss


if __name__ == "__main__":
    device = torch.device('cpu')
    
    model = sampleModel().to(device)

    data = []
    for i in range(0,500):
        data.append(np.random.randint(0, 20, size=30))
    
    inputx = torch.LongTensor(data).to(device)
    labels = torch.LongTensor( [0]*250 + [1]*250 ).to(device)
    
    loss = model(inputx, labels)
    print(loss)
    loss.backward()