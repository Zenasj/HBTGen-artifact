import torch.nn as nn

import torch
import torch._dynamo as dynamo

def init_model():
    return torch.nn.Linear(10, 10)

def generate_data(b):
    return torch.randn(b, 10)

@dynamo.optimize("inductor")
def eval(model, inp):
    model(inp)
    # return model(inp) works

eval(init_model(), generate_data(16))