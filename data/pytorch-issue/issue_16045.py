import torch.nn as nn

def train_epoch(model:nn.Module, dl:DataLoader, opt:optim.Optimizer, loss_func:LossFunction)->None:
    "Simple training of `model` for 1 epoch of `dl` using optim `opt` and loss function `loss_func`."
    model.train()
    for xb,yb in dl:
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

if not (target.size() == input.size()):
    # first handle the one-dimensional cases 
    if target.ndimension() == 1:
        target = target.unsqueeze(1)
    elif input.ndimension() == 1:
        input = input.unsqueeze(1)
    else:
        warnings.warn("Using a target size ({}) that is different to the input size ({})."
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
        
if input.numel() != target.numel():
    raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                     "!= input nelement ({})".format(target.numel(), input.numel()))

model.train()
for xb,yb in dl:
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()