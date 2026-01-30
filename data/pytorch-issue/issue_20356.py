import torch.nn as nn

import torch

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

my_values = {
    'a': torch.ones(2, 2),
    'b': torch.ones(2, 2) + 10,
    'c': 'hello',
    'd': 6
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
container.save("container.pt")

import io
f = io.BytesIO()
torch.save(x, f, _use_new_zipfile_serialization=True)
# send f wherever

x = torch.load(f)

import torch
torch.load("x.zip")

import io

import torch


def save_tensor(device):
    my_tensor = torch.rand(3, 3).to(device);
    print("[python] my_tensor: ", my_tensor)
    f = io.BytesIO()
    torch.save(my_tensor, f, _use_new_zipfile_serialization=True)
    with open('my_tensor_%s.pt' % device, "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())


if __name__ == '__main__':
    save_tensor('cpu')