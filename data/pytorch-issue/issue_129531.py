3
# Owner(s): ["oncall: pt2"]
import sys
import textwrap
import unittest

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor import config
from torch._inductor.codecache import HalideCodeCache
from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import parallel_num_threads

from torch.testing._internal.common_utils import IS_CI, IS_MACOS, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CPU
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    import halide

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


make_halide = config.patch(
    {
        "cpu_backend": "halide",
        "cuda_backend": "halide",
        "fallback_random": True,  # TODO(jansel): support random
    }
)

def test_manual_schedule():

    fn = HalideCodeCache.generate_halide(
        HalideMeta(
            argtypes=[
                HalideInputSpec(
                    ctype="float*",
                    name="in_ptr0",
                    shape=["1024L"],
                    stride=["1L"],
                    offset="0",
                    alias_of=None,
                ),
                HalideInputSpec(
                    ctype="float*",
                    name="in_ptr1",
                    shape=["1024L"],
                    stride=["1L"],
                    offset="0",
                    alias_of=None,
                ),
                HalideInputSpec(
                    ctype="float*",
                    name="out_ptr0",
                    shape=["1024L"],
                    stride=["1L"],
                    offset="0",
                    alias_of=None,
                ),
            ],
            target="host-cuda-cuda_capability_86-user_context-strict_float-no_runtime-no_asserts-debug",
            scheduler=None,
            cuda_device=0,
        ),
        textwrap.dedent(
            """
            import halide as hl

            @hl.generator(name="kernel")
            class Kernel:
                in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                def generate(g):
                    in_ptr0 = g.in_ptr0
                    in_ptr1 = g.in_ptr1
                    out_ptr0 = g.out_ptr0
                    xindex = hl.Var('xindex')
                    x0 = xindex
                    tmp0 = hl.Func()
                    tmp0[xindex] = in_ptr0[x0]
                    tmp1 = hl.Func()
                    tmp1[xindex] = in_ptr1[x0]
                    tmp2 = hl.Func()
                    tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                    out_ptr0[x0] = tmp2[xindex]

                    assert not g.using_autoscheduler()
                    i = hl.Var()
                    j = hl.Var()
                    out_ptr0.compute_root()
                    out_ptr0.split(xindex, i, j, 32)
                    out_ptr0.parallel(i)
                    out_ptr0.vectorize(j)
                    tmp2.compute_at(out_ptr0, i)
                    tmp2.store_at(out_ptr0, i)
                    tmp1.compute_inline()

            if __name__ == '__main__':
                hl.main()
                """,
        )
    )

    a = torch.randn(1024).to(device='cuda:0')
    b = torch.randn(1024).to(device='cuda:0')
    c = torch.randn(1024).to(device='cuda:0')
    torch.cuda.set_device(0)
    stream0 = get_raw_stream(0)
    fn(a, b, c, stream0)
    torch.testing.assert_allclose(c, a + b)
    del a, b, c
    
if __name__ == "__main__":
    test_manual_schedule()