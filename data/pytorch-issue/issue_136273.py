import torch
import os
import pathlib
from ctypes import cdll

if int(os.getenv("NO_LOAD", "0")) == 0:
    lib = pathlib.Path(os.getenv("LIB",""))
    if not lib.is_file():
        from torch._inductor.codecache import CppCodeCache
        path = CppCodeCache.load("")._name
        raise ValueError(f"export LIB=\"{path}\" and re-run.")

    cdll.LoadLibrary(lib)

device="cpu"
dtype=torch.complex128
finfo = torch.finfo(dtype)
nom = torch.tensor([complex(finfo.min / 2, finfo.min / 2)], dtype=dtype, device=device)
denom = torch.tensor([complex(finfo.min / 2, finfo.min / 2)], dtype=dtype, device=device)
res = (nom / denom).item()
# Expected complex(1.0, 0.0)
assert res.real > 0.5, res
print("OK!")