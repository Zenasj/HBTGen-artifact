import torch
import torch.nn.functional as F

with torch.backends.cuda.sdpa_kernel(backends=[SDPBackend.MATH]):
    F.sdpa(...)