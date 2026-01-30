import torch.nn as nn

import torch

def capture(fn):
    def inner(*args):
        gm = None
        actual_args = None
        kwargs = None

        def backend(gm_, args_, **kwargs_):
            nonlocal gm
            nonlocal actual_args
            nonlocal kwargs
            gm = gm_
            actual_args = args_
            kwargs = kwargs_
            return gm

        _ = torch.compile(fn, fullgraph=True, backend=backend)(*args)
        return gm, actual_args, kwargs

    return inner

model = torch.nn.Linear(16, 16, device='cuda')
inp = torch.randn(16, 16, device='cuda')

with torch.no_grad():
    gm, args, kwargs = capture(model)(inp)
    assert not kwargs
    compiled_artifact = torch._inductor.standalone_compile(gm, args)
    path = 'tmp_cache_dir'
    format = 'unpacked'
    compiled_artifact.save(path=path, format=format)
    loaded = torch._inductor.CompiledArtifact.load(path=path, format=format)
    compiled_out = loaded(inp)