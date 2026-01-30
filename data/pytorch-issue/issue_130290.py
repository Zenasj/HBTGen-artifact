import torch

@torch.compile
@torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
def minimal_repro(
    tensor: torch.Tensor,
    mapping: torch.Tensor,
) -> torch.Tensor:
    xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
    indices = torch.argwhere(xx == yy)

    mapped_values = torch.zeros_like(tensor)
    mapped_values[indices[:, 1]] = indices[:, 0]

    return mapped_values

def test_minimal_repro() -> None:
    tensor = torch.tensor([1, 2, 3, 5, 6, 7])
    mapping = torch.tensor([0, 3, 4, 5, 7])

    mapped_tensor = minimal_repro(tensor, mapping)
    torch.testing.assert_close(mapped_tensor, torch.tensor([0, 0, 1, 3, 0, 4]))

def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0), xmask)
    tmp6 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([XBLOCK], 6, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 6)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 6")
    tl.store(out_ptr0 + (tl.broadcast_to(tmp4, [XBLOCK])), tmp6, xmask)