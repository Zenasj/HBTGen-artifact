import torch.nn as nn

# Most of the code has been adapted from a script authored by leslie-fang-intel
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch._inductor.config as config
from torchao.quantization import quant_api
from torchao.utils import unwrap_tensor_subclass

config.freezing = True
config.cpp_wrapper = True

M=1
N=4096
K=4096

class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(K, N)

    def forward(self, x, x2):
        tmp = self.linear(x)
        return tmp
    
if __name__ == "__main__":
    m = Model().eval()
    input = torch.randn(M, K)
    input2 = torch.randn(M, N)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
        quant_api.quantize_(m, quant_api.int8_weight_only(), set_inductor_config=False)
        cm = torch.compile(m)
        res = cm(input, input2)
        print("---- benchmark Inductor WOQ INT8 ----", flush=True)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            cm(input, input2)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100), flush=True)