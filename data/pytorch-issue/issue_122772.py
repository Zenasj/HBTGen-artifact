import torch
import torch.nn as nn

# The input is a tuple of tensors with shapes: (7,), (3,), (3,), (12,4,5), (12,5,6), (12,4,6), (5,), (1,), (1,)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        (input_topk, out_topk_values, out_topk_indices,
         input_bmm1, input_bmm2, out_bmm,
         input_max, out_max_values, out_max_indices) = inputs

        # Perform topk with k=3
        torch.topk(input_topk, 3, out=(out_topk_values, out_topk_indices))

        # Perform bmm
        torch.bmm(input_bmm1, input_bmm2, out=out_bmm)

        # Perform max with keepdim=True
        torch.max(input_max, 0, keepdim=True, out=(out_max_values, out_max_indices))

        return out_topk_values, out_bmm, out_max_values

def my_model_function():
    return MyModel()

def GetInput():
    # Topk inputs (second case causing error)
    input_topk = torch.arange(1., 8.)  # shape (7,)
    out_topk_values = torch.empty(3)
    out_topk_indices = torch.empty(3, dtype=torch.long)

    # Bmm inputs (second case)
    input_bmm1 = torch.randn(12, 4, 5)
    input_bmm2 = torch.randn(12, 5, 6)
    out_bmm = torch.empty(12, 4, 6)

    # Max inputs (second case)
    input_max = torch.randn(5)
    out_max_values = torch.empty(1)
    out_max_indices = torch.empty(1, dtype=torch.long)

    return (input_topk, out_topk_values, out_topk_indices,
            input_bmm1, input_bmm2, out_bmm,
            input_max, out_max_values, out_max_indices)

