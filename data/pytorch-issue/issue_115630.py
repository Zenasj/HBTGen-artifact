import torch.nn as nn

import os

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

with open("MyModule.pt","wb") as f:
    f.write(b"header")
    torch.save(MyModule(),f)


with open("MyModule.pt","rb") as f:
    f.read(6)
    orig = f.tell()
    f.seek(orig,os.SEEK_END) # happy to going after end !
    wrong_size = f.tell()-orig
    f.seek(0,os.SEEK_END)
    good_size = f.tell()-orig
    print(f"size:{wrong_size} != {good_size}") # wrong size calculated

with open("MyModule.pt","rb") as f:
    f.read(6)
    torch.load(f)