import torch.nn as nn

import torch
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
lr = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1.0)

tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    torch.save(lr.state_dict(), f)

import torch
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
lr = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1.0)

tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    state_dict = lr.state_dict()
    print(state_dict["_scale_fn_ref"])
    torch.save(state_dict["_scale_fn_ref"], f)

import torch
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)

# no custom scale function so it'll use WeakMethod by default
lr = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1.0)

# instantiate the WeakMethod in the lr scheduler object into the custom scale function attribute
lr._scale_fn_custom = lr._scale_fn_ref()

# remove the reference so there are no more WeakMethod references in the object
lr._scale_fn_ref = None

# now we can successfully pickle the lr scheduler
tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    state_dict = lr.state_dict()
    torch.save(state_dict, f)