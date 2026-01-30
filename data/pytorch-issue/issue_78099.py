import random

import numpy as np
import pandas as pd
import torch
import sklearn
import matplotlib.pyplot as plt

# set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

np.random.seed(42)

a = torch.tensor(np.random.rand(10000, 4200), dtype=torch.float32).to(device)
print(f"a shape: {a.shape}")

U, S, Vh = torch.linalg.svd(a, full_matrices=False)

aa = U @ torch.diag(S) @ Vh

torch.linalg.norm(aa - a)

import torch
import numpy as np

device = "mps"
print(f"Using device: {device}")
np.random.seed(42)
a = torch.tensor(np.random.rand(10000, 4200), dtype=torch.float32)
print(f"a shape: {a.shape}")

U, S, Vh = torch.linalg.svd(a.to(device), full_matrices=False)
aa = U @ torch.diag(S) @ Vh
cpu_aa = U.cpu() @ torch.diag(S.cpu()) @ Vh.cpu()

print(torch.linalg.norm(aa - a.to(device)))
print(torch.linalg.norm(cpu_aa - a.cpu()))

print(U)
print(U.cpu())
print(U.cpu().clone().to(device))
print(torch.tensor(U.cpu().detach().numpy()).to(device))
print(torch.tensor(U.cpu().detach().numpy().copy()).to(device))

mps_U = torch.tensor(U.cpu().detach().numpy().copy()).to(device)
mps_S = torch.diag(torch.tensor(S.cpu().detach().numpy().copy()).to(device))
mps_Vh = torch.tensor(Vh.cpu().detach().numpy().copy()).to(device)

print(torch.linalg.norm(mps_U @ mps_S @ mps_Vh - a.to(device)))