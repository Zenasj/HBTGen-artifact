import logging
import torch
from torch_frame.data import MultiNestedTensor
torch._logging.set_logs(dynamo=logging.INFO)
torch._dynamo.config.verbose = True

def f(tensor: MultiNestedTensor):
    offset = tensor.offset[:: tensor.num_cols]

feat = MultiNestedTensor.from_tensor_mat(
    [
        [
            torch.randint(1, (2,)),
            torch.randint(1, (2,)),
        ],
    ],
)
assert feat.num_cols == 2
c_f = torch.compile(f, dynamic=True, backend="eager")
c_f(feat)