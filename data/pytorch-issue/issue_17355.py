import torch
import torch.nn as nn

print(torch.log_softmax.__doc__)
# None

print(torch.nn.functional.log_softmax.__doc__)
# Correctly prints documentation.