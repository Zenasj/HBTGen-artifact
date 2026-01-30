import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

lin1 = nn.Linear(2,2, bias=False)
fully_shard(lin1)
print(f"BEFORE, the param has dtype fp32 {lin1.weight=} {lin1.weight._spec.tensor_meta}")
t = torch.zeros_like(lin1.weight, dtype=torch.bfloat16)
print(f"AFTER, the param has dtype bf16????? {lin1.weight=} {lin1.weight._spec.tensor_meta}")