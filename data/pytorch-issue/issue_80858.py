import torch

from torch.testing._internal.composite_compliance import generate_cct
from torch.testing import make_tensor

CCT = generate_cct()

# Passes!
t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=False)
torch.clone(CCT(t))

# Passes!
t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=False, requires_grad=True)
torch.clone(CCT(t))

# Passes!
t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=True)
torch.clone(CCT(t))

# Fails!
# RuntimeError: This operator is not Composite Compliant: the stride of the tensor was modified directly without going through the PyTorch dispatcher.
t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=True, requires_grad=True)
torch.clone(CCT(t))

import torch

from torch.testing._internal.composite_compliance import generate_cct
from torch.testing import make_tensor

CCT = generate_cct()

t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=True, requires_grad=False)
cct = CCT(t)

torch.empty_like(cct)  # Fails

import torch

from torch.testing._internal.composite_compliance import generate_cct
from torch.testing import make_tensor

CCT = generate_cct()

t = make_tensor((5, 5, 5), device='cpu', dtype=torch.float, noncontiguous=True, requires_grad=True)
cct = CCT(t)
assert cct.elem.stride() == cct.stride()  # Fails when requires_grad=True and noncontiguous=True