import numpy as np
import torch
print(np.__version__) #1.24.3
print(torch.__version__) #2.0.0
# print(torch.__config__.show())

x = torch.tensor([1], dtype=torch.complex128)
print(torch.dot(x, x)) #tensor(0.+0.j, dtype=torch.complex128)
# expected: tensor(1, dtype=torch.complex128)

import torch
print(torch.__version__)
x = torch.tensor([1], dtype=torch.complex128)
print(torch.dot(x, x))

# main-no-np.py
import torch
x = torch.tensor([1], dtype=torch.complex128)
print(torch.dot(x, x))

# main-with-np.py
import numpy as np
import torch
x = torch.tensor([1], dtype=torch.complex128)
print(torch.dot(x, x))