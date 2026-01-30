import torch
import pdb

class _OpenRegNewOne:
    pass

torch.utils.rename_privateuse1_backend("new_one")
torch._register_device_module('new_one', _OpenRegNewOne())
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)

a1 = torch.Tensor(3,4).to("new_one")

a1 = torch.Tensor(3,4).to("new_one")

a1 = torch.Tensor(1,2).to('new_one')