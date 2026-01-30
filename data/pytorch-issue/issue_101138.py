import torch

import subprocess
import os
from torch.utils import cpp_extension

print(subprocess.check_output(['where', 'nvcc'], stderr=open(os.devnull, 'w')))
print(cpp_extension._find_cuda_home())

from torch.utils import cpp_extension
cpp_extension._check_cuda_version('cl', '19.29.30148')