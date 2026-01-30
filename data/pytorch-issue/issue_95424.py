import torch.nn as nn

import torch
import torch._dynamo
import numpy as np

class MyModule(torch.nn.Linear):

    def __init__(self):
        super().__init__(np.array([4, 4, 1]).prod(), np.array([4, 4, 1]).prod())

    def forward(self, x):
        # return x.reshape(1, self.in_features). # This line passed
        return x.reshape(self.in_features)

x = torch.rand([4, 4])
model = MyModule()
print(model(x))
opt_model = torch._dynamo.optimize("eager")(model)
print(opt_model(x))

self.in_features

UnspecializedPythonVariable

FakeTensor

reshape

FakeTensor