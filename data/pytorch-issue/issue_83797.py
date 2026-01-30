import torch.nn as nn

def __getitem__(self, idx: int) -> Union[Module, 'ModuleList']:
    if isinstance(idx, slice):
        return self.__class__(list(self._modules.values())[idx])
    else:
        return self._modules[self._get_abs_string_index(idx)]

import torch.nn

module_list = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
module_sublist = module_list[:-1]  # Pyright throws an error