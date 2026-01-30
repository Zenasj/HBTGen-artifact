import torch
from torch import optim

params = (torch.empty(1),)

adadelta = optim.Adadelta(params)
adagrad = optim.Adagrad(params)
adam = optim.Adam(params)  # checks fine
adamax = optim.Adamax(params)
adamw = optim.AdamW(params)  # checks fine
asgd = optim.ASGD(params)
lbfgs = optim.LBFGS(params)
rmsprop = optim.RMSprop(params)
rprop = optim.Rprop(params)
sgd = optim.SGD(params, lr=1.0)  # checks fine
sparse_adam = optim.SparseAdam(params)