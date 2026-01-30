import torch
print("torch.__version__:", torch.__version__)
torch._C._host_emptyCache()
print("ok")