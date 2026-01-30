import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.parameter_dict = nn.ParameterDict({"foo": nn.Parameter(torch.zeros(1, 1, 128))})
        self.parameter = self.parameter_dict["foo"]

    def forward(self, x):
        return self.parameter_dict["foo"] + x # this breaks
        # return self.parameter + x  #this works fine


model = Model()
model = torch.compile(model, backend="eager")
x = torch.randn(1, 1, 128)

model(x)