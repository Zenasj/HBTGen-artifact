import torch
import torch.nn as nn

inner_module = torch.jit.load("inner.pt")
wrapped = Wrapper(inner_module)  # wrap the loaded ScriptModule in an nn.Module
torch.jit.script(wrapped)