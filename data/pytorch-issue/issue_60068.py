import torch
model1 = torch.load("folder1/model1.pt")
model2 = torch.load("folder2/model2.pt")
print(model1)
print(model2)

import torch
model1 = torch.load("../folder1/model1.pt")
model2 = torch.load("../folder2/model2.pt")
print(model1)
print(model2)