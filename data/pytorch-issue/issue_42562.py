import torch.nn as nn

from typing import Dict, List, Optional
import torch

class TypedDataDict(object):
    def __init__(
        self,
        str_to_tensor: Optional[Dict[str, torch.Tensor]] = None,
        str_to_list_of_str: Optional[Dict[str, List[str]]] = None
    ):
        self.str_to_tensor = str_to_tensor
        self.str_to_list_of_str = str_to_list_of_str

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input: torch.Tensor):
    return TypedDataDict()

m = TestModule()
torch.jit.script(m)