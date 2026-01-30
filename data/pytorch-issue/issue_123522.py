import torch.nn as nn

import torch
from torchao.quantization import quant_api
 
class M(torch.nn.Module):
    def __init__(
        self,
        use_bias,
        post_op_algo,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, use_bias)
        self.linear2 = torch.nn.Linear(16, 16, use_bias)
 
    def forward(self, x):
        temp = self.linear1(x)
        temp = self.linear2(temp)
 
        return temp
 
 
if __name__ == "__main__":
    bias = False
    mod = M(bias, "none").eval()
 
    v = torch.randn((3, 16))
    inputs = (v,)
 
    quant_api.change_linear_weights_to_int8_woqtensors(mod)
 
    ref_res = mod(*inputs)
 
    with torch.no_grad(), torch.cpu.amp.autocast():
        compiler_mode = torch.compile(mod)
        _ = compiler_mode(*inputs)
        output = compiler_mode(*inputs)
 
        print(torch.allclose(ref_res, output), flush=True)