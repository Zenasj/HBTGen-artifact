py
import torch.nn as nn
import torch


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("some_state", None)


module_original = MyModule()
module_original.some_state = torch.tensor(42)

# later when loading checkpoint
module_copy = MyModule()
module_copy.load_state_dict(module_original.state_dict())

py
for name, buf in self._buffers.items():
    if buf is not None and name not in self._non_persistent_buffers_set:
        destination[prefix + name] = buf if keep_vars else buf.detach()