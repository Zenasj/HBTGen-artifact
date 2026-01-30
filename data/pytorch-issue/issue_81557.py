import torch
import torch.nn as nn

def repro():
    model = torch.nn.Conv1d(1, 128, 3)
    a = torch.ones(128, 176, 1).permute(0, 2, 1)
    out = model(a)  # pass

    a_mps = a.to("mps")
    model = model.to("mps")
    out = model(a_mps)  # fail