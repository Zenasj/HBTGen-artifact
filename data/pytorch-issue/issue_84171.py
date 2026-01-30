import torch.nn as nn
import numpy as np

import torch
import numpy
import pickle as pkl

layerinp=torch.Tensor(pkl.load(open("torch-conv-data.pkl", "rb")))
layerinp=layerinp.to("cuda")

model = torch.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
model.to("cuda")
model.eval()
with torch.no_grad():
    output=model(layerinp).cpu().data.numpy()

print("pt output", output[0][0][0])