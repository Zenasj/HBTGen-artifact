import torch
import triton
import triton.language as tl

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def get_default_config():
    config = triton.Config({'BLOCK_SIZE':1024}, num_warps=4, num_stages=2, pre_hook=init_to_zero("output_ptr"))
    return [config]

@triton.autotune(configs = get_default_config(), key=['n_elements'],)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)  

    block_start = pid * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)
    mask        = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.atomic_add(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output     = torch.ones(x.shape, device=x.device, dtype=x.dtype)
    n_elements = output.numel()
    grid       = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements)
    return output


x = torch.ones((4096,), device='cuda:0', dtype=torch.float16)
y = torch.ones((4096,), device='cuda:0', dtype=torch.float16)

assert add(x, y).mean() == 2, "Problem with add kernel"

for mode in ['reduce-overhead', 'max-autotune-no-cudagraphs']:
    add_compiled = torch.compile(add, mode=mode, fullgraph=True)  
    val = add_compiled(x, y).mean().item()  
    if(val != 2):
        print("pre_hook ignored with " + mode, 'expected 2, but found ' + str(val))


# pre_hook ignored with reduce-overhead expected 2, but found 3.0
# pre_hook ignored with max-autotune-no-cudagraphs expected 2, but found 3.0