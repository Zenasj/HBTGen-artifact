import torch

@torch.compile(backend="eager")
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))

def compiled_code(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        # Dynamo will call compiled code for __resume_at_30_1 , but debugpy directly run the bytecode
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)

import depyf
with depyf.prepare_debug(toy_example, "./dump_src_dir"):
    for _ in range(100):
        toy_example(torch.randn(10), torch.randn(10))

with depyf.debug():
    toy_example(torch.randn(10), torch.randn(10))