import torch.nn as nn

import torch
import torch._inductor

# Taken from test_config_option_dont_assume_alignment_cuda

class M(torch.nn.Module):
    def forward(self, x):
        return x.sin() + x.cos()

N = 64 * 64 * 64 + 64
dtype = torch.float32

arg = torch.randn(N, dtype=dtype, device='cuda')
args = (arg,)

m_arg = torch.zeros(N + 1, dtype=dtype, device='cuda')
m_arg = m_arg[1:]
m_arg.copy_(arg)
m_args = (m_arg,)

fn = M()
opt_fn = torch.compile()(fn)
print(opt_fn(*args))
# This triggers IMA if you stubbed out copy_misaligned_inputs
#print(opt_fn(*m_args))

fn = torch._inductor.aoti_compile_and_package(torch.export.export(M(), args))
print(fn)

model = torch._C._aoti.AOTIModelPackageLoader(fn, "model")

with torch.profiler.profile() as p:
    r = model.run(m_args)

print(r)
# Expect to see copy_ here
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))