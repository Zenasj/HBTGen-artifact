...
torch.save(net, "dump.pt")
load_net = torch.load("dump.pt")
...

import torch
import pretrainedmodels

model_name = "alexnet" # for alexnet
model_args = {} # for alexnet
net = getattr(pretrainedmodels.models, model_name)(**model_args)


torch.save(net.state_dict(), "dump.pt")
load_net_sd = torch.load("dump.pt")
net.load_state_dict(load_net_sd)