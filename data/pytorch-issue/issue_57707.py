import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import skipCUDAIfNoMagma
names = set()
names_no_bf16 = set()
for o in op_db:
    if ('linalg' not in o.name and 'fft' not in o.name):
        if (o.decorators is None or skipCUDAIfNoMagma not in o.decorators):
            names.add(o.name)
            if not torch.bfloat16 in o.dtypesIfCUDA and torch.float in o.dtypesIfCUDA:
                names_no_bf16.add(o.name)
for n in sorted(names_no_bf16):
    print(n)
print(len(names), len(names_no_bf16))