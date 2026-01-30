import torch.nn as nn

import torch

r = torch.nn.parallel.replicate(torch.nn.Linear(2, 2).to(0), devices=[0, 1], detach=True)
torch.save(r, "./tmp_module.pt")

import torch

r = torch.nn.parallel.replicate(torch.nn.Linear(2, 2).to(0), devices=[0, 1], detach=False)
torch.save(r, "./tmp_module.pt")