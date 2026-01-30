import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x[1].fill_diagonal_(0)   # this check size failed

device = torch.device("cpu")
model = Model()
model.to(device)

torch._dynamo.reset()
compiled_model = torch._dynamo.optimize("inductor")(model)

arg = [torch.rand([4, 1, 1])]
compiled_model(*arg)