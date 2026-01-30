python
import torch

x = torch.randn(10, device="xpu", dtype=torch.bfloat16)

def profile_and_check(fn, x, kwargs):
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
        fn(x, **kwargs, dtype=torch.float)
    # check that profiler returned some events
    assert "aten::linalg_vector_norm" in (e.name for e in p.events())
    # test that there was no explicit copy
    assert "aten::to" not in (e.name for e in p.events())

# torch.linalg.vector_norm
profile_and_check(torch.linalg.vector_norm, x, {})