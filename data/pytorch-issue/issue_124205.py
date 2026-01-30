from torch.nested._internal.nested_tensor import NestedTensor
import torch

@torch.compile
@torch._dynamo.config.patch(error_on_recompile=True)
def sample_fun(njt: NestedTensor) -> NestedTensor:
    njt = njt.clamp(0.1, 0.5)
    # if this is NOT specialized - this should lead to second printed output be larger than first
    njt *= njt._max_seqlen
    return njt

def create_njt(el_per_row):
    torch.manual_seed(0)
    njt = NestedTensor(
        values=torch.randn(10 * el_per_row, device="cuda"),
        offsets=torch.arange(11, device="cuda") * el_per_row,
    )
    # This sets _max_seqlen in cache
    print(njt._max_seqlen)
    return njt

with torch.inference_mode():
    # This works
    print(sample_fun(create_njt(el_per_row=1)).values())
    # But this printed output should be 2x as big - and it is NOT as max_seqlen is specialized in generated code.
    print(sample_fun(create_njt(el_per_row=2)).values())