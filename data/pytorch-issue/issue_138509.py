import torch
import triton
import triton.language as tl


def _dtype_min_max(dtype: torch.dtype) -> tuple[tl.constexpr, tl.constexpr]:
    info = torch.finfo(dtype)
    return tl.constexpr(info.min), tl.constexpr(info.max)


_FLOAT8_MIN, _FLOAT8_MAX = _dtype_min_max(torch.float8_e4m3fn)
_FLOAT16_MIN, _FLOAT16_MAX = _dtype_min_max(torch.float16)
_BFLOAT16_MIN, _BFLOAT16_MAX = _dtype_min_max(torch.bfloat16)


@triton.jit
def scale_and_clamp(x, scale, dtype):
    if dtype == tl.float8e4nv:
        clamp_min = _FLOAT8_MIN
        clamp_max = _FLOAT8_MAX
    elif dtype == tl.float16:
        clamp_min = _FLOAT16_MIN
        clamp_max = _FLOAT16_MAX
    elif dtype == tl.bfloat16:
        clamp_min = _BFLOAT16_MIN
        clamp_max = _BFLOAT16_MAX
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")
    return tl.clamp(x * scale, clamp_min, clamp_max).to(dtype)


@triton.jit
def _scaled_cast_kernel(
    o_ptr,
    x_ptr,
    scale_ptr,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(axis=0)
    offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    result = scale_and_clamp(x, scale, o_ptr.dtype.element_ty)
    tl.store(o_ptr + offsets, result, mask=mask)


def scaled_cast(
    x: torch.Tensor,
    dtype: torch.dtype,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Scaled type conversion kernel."""

    d = x.numel()
    o = torch.empty((d,), dtype=dtype, device=x.device)

    def grid(meta):
        return (triton.cdiv(d, meta["BLOCK_SIZE"]),)

    _scaled_cast_kernel[grid](
        o_ptr=o,
        x_ptr=x,
        scale_ptr=scale,
        d=d,
        BLOCK_SIZE=1024,
    )

    return o.reshape(x.shape)


def test_scaled_cast() -> None:
    device = torch.device(0)
    x = torch.rand((8,), dtype=torch.float16, device=device)
    scale = torch.tensor(8, dtype=torch.float32, device=device)
    scaled_cast_fn = torch.compile(scaled_cast, fullgraph=True)
    result = scaled_cast_fn(x, torch.float8_e4m3fn, scale)
    torch.testing.assert_close(
        x.to(torch.float32),
        result.to(torch.float32) * scale.reciprocal(),
        atol=1e-1,
        rtol=1e-3,
    )


if __name__ == "__main__":
    test_scaled_cast()

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'kernel_name': '_scaled_cast_kernel_0', 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'o_ptr': '*fp8e4nv', 'x_ptr': '*fp16', 'scale_ptr': '*fp32', 'd': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {'BLOCK_SIZE': 1024}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _scaled_cast_kernel(
    o_ptr,
    x_ptr,
    scale_ptr,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(axis=0)
    offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    result = scale_and_clamp(x, scale, o_ptr.dtype.element_ty)
    tl.store(o_ptr + offsets, result, mask=mask)

@triton.jit
def scale_and_clamp(x, scale, dtype):
    if dtype == tl.float8e4nv:
        clamp_min = _FLOAT8_MIN
        clamp_max = _FLOAT8_MAX
    elif dtype == tl.float16:
        clamp_min = _FLOAT16_MIN
        clamp_max = _FLOAT16_MAX
    elif dtype == tl.bfloat16:
        clamp_min = _BFLOAT16_MIN
        clamp_max = _BFLOAT16_MAX
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")
    return tl.clamp(x * scale, clamp_min, clamp_max).to(dtype)

_FLOAT8_MIN = constexpr[-448.0]

_FLOAT8_MAX = constexpr[448.0]

_FLOAT16_MIN = constexpr[-65504.0]

_FLOAT16_MAX = constexpr[65504.0]

_BFLOAT16_MIN = constexpr[-3.3895313892515355e+38]

_BFLOAT16_MAX = constexpr[3.3895313892515355e+38]