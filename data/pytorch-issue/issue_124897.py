import torch.nn.functional as F

import torch
import os
print(torch.__version__)
x=open(os.path.join(os.path.dirname(torch.__file__), "_VF.pyi"), "rb").read()
y=x.decode("utf-8")
print(len(x), len(y))