import torch.nn as nn

import torch
from typing import Dict, Optional

class TMP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                in_batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        self.dropout_modality(in_batch)
        return torch.tensor(1)

    @torch.jit.ignore
    def dropout_modality(self,
                         in_batch: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        return in_batch

tmp_script = torch.jit.script(TMP())