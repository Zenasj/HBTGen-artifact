import random

import torch
import os


x= torch.randint(0, 100, size=(10000,8), dtype=torch.long)

y1 = x.narrow(dim=0, start=0, length=10)
torch.save({"y": y1},"y1.pth")

y2 = x.narrow(dim=0, start=0, length=10).detach().clone()
torch.save({"y": y2},"y2.pth")


loaded_y1 = torch.load("y1.pth")["y"]
loaded_y2 = torch.load("y2.pth")["y"]

if torch.equal(loaded_y1, loaded_y2):
    print("loaded_y1==loaded_y2")

if torch.equal(loaded_y1, y1):
    print("loaded_y1==y1")

print("size y1.pth (in Bytes)=", os.stat("y1.pth").st_size)
print("size y2.pth (in Bytes)=", os.stat("y2.pth").st_size)

import numpy as np
import os

x = np.random.randint(0, 100, size=(10000,8), dtype=np.int64)

y1 = x[0:10]
np.save("y1.npy", y1)

y2 = np.copy(x[0:10])
np.save("y2.npy", y2)

loaded_y1 = np.load("y1.npy")
loaded_y2 = np.load("y2.npy")

if np.array_equal(loaded_y1, loaded_y2):
    print("loaded_y1==loaded_y2")

if np.array_equal(loaded_y1, y1):
    print("loaded_y1==y1")

print("size y1.npy (in Bytes)=", os.stat("y1.npy").st_size)
print("size y2.npy (in Bytes)=", os.stat("y2.npy").st_size)