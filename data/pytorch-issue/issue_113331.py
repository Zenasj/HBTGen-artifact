import torch.nn as nn

import torch
import math
from torch._export import capture_pre_autograd_graph

class M(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, scores, mask):
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)
        return scores

if __name__ == "__main__":
    tensor_cpu = torch.randn(2, 4)
    mask_cpu = torch.BoolTensor(
        [[False,  True, False, False],
        [False, False, False, False]]
    )

    m = M().eval()
    # res_ref = m(tensor_cpu, mask_cpu)
    # print("res_ref is: {}".format(res_ref), flush=True)

    exported_model = capture_pre_autograd_graph(
        m,
        (tensor_cpu, mask_cpu),
    )
    print(exported_model, flush=True)
    optimized_model = torch.compile(exported_model)
    optimized_model(tensor_cpu, mask_cpu)