import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        ...
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, inputs):
        # inputs: [B, C, H, W]
        outputs = self.pool(inputs)
        # outputs: [B, C, 1, W]
        return outputs

_size_any_opt_t = _scalar_or_tuple_any_t[Optional[int]]
_size_1_opt_t = _scalar_or_tuple_1_t[Optional[int]]
_size_2_opt_t = _scalar_or_tuple_2_t[Optional[int]]
_size_3_opt_t = _scalar_or_tuple_3_t[Optional[int]]
_size_4_opt_t = _scalar_or_tuple_4_t[Optional[int]]
_size_5_opt_t = _scalar_or_tuple_5_t[Optional[int]]
_size_6_opt_t = _scalar_or_tuple_6_t[Optional[int]]