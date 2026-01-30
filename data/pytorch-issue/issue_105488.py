import torch

data = [{"1": 1}]

with open("H:\\a.txt", "w") as f:
    f.write("test")

torch.save(data, "H:\\b.ckpt")

import torch

data = [{"1": 1}]

with open("H:/a\\a.txt", "w") as f:
    f.write("test")

torch.save(data, "H:/a\\b.ckpt")

import torch

data = [{"1": 1}]

with open("H:/a/a.txt", "w") as f:
    f.write("test")

torch.save(data, "H:/a/b.ckpt")

import torch

data = [{"1": 1}]

with open("H:/a/a\\a.txt", "w") as f:
    f.write("test")

torch.save(data, "H:/a/a\\b.ckpt")

import torch

data = [{"1": 1}]

with open("H:\\a\\a.txt", "w") as f:
    f.write("test")

torch.save(data, "H:\\a\\b.ckpt")