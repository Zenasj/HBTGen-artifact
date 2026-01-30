import torch
for dtype in [torch.float64, torch.float16]:
    def fn():
        return torch.normal(2, 3, (100, 100), dtype=dtype, device="cpu")
    
    assert torch.compile(fn, backend="eager")().dtype ==  dtype # success
    assert torch.compile(fn, backend="aot_eager")().dtype == dtype  # success
    assert torch.compile(fn, backend="aot_eager_decomp_partition")().dtype == dtype  # fail
    assert torch.compile(fn, backend="inductor")().dtype == dtype  # fail