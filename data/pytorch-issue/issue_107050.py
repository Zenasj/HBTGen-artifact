import torch
from torch.func import vmap

def minor_repro():
    # number = 0.5
    # number = torch.randn(())
    number = torch.randn(()).item()  # Why does this show up as a tensor?

    # Why does shape matter?
    # x = torch.randn(3, 3, 3, device='cpu') # Works!

    x = torch.randn(3, 3, device='cpu') # Doesn't Work and crashes with Please convert all Tensors to FakeTensors first

    # Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'.
    # Found in aten.add.Tensor(FakeTensor(..., size=(3,)), tensor(-0.8092, size=()))
    vmap(vmap(lambda t: torch.add(t, number)))(x)

# minor_repro()
torch.compile(minor_repro, backend='eager')()