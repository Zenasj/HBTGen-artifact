import torch
import torch.nn as nn


def train(netD, netG, x):
    # train netD
    fake = netG(x)
    outD_only = netD(fake.detach())
    outD_only.mean().backward()
    
    # confirm the backward call does not touch netG
    print(["{}, {}".format(name, p.grad) for name, p in netG.named_parameters()])
    
    # train netG
    outD_attached = netD(fake)
    outD_attached.mean().backward()
    
    # confirm netG now has valid gradients
    print(["{}, {}".format(name, p.grad) for name, p in netG.named_parameters()])


## Eager
# setup
netG = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1))

netD = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1))

x = torch.randn(1, 1)

train(netD, netG, x)
# ['0.weight, None', '0.bias, None', '2.weight, None', '2.bias, None']
# ['0.weight, tensor([[6.7547e-05]])', '0.bias, tensor([0.0002])', '2.weight, tensor([[0.0008]])', '2.bias, tensor([0.0009])']

   
## Compile
# setup
netG = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1))

netD = nn.Sequential(
    nn.Linear(1, 1),
    nn.ReLU(),
    nn.Linear(1, 1))

x = torch.randn(1, 1)
train_compiled = torch.compile(train, mode="default")

train_compiled(netD, netG, x)
# ['0.weight, tensor([[0.]])', '0.bias, tensor([0.])', '2.weight, tensor([[0.]])', '2.bias, tensor([0.])'] # grads seems to be already created
# RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.