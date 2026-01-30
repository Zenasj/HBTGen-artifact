import torch
import triton
import triton.language as tl


@triton.jit
def addition_kernel(x_ptr, y_ptr, out_ptr, stride_x0, stride_x1, stride_y0, stride_y1, stride_o0, stride_o1, SIZE_0: tl.constexpr, SIZE_1: tl.constexpr, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr):
    for i0 in range(0, SIZE_0, BLOCK_SIZE_0):
        off0 = tl.arange(0, BLOCK_SIZE_0) + i0
        mask0 = off0 < SIZE_0
        for i1 in range(0, SIZE_1, BLOCK_SIZE_1):
            off1 = tl.arange(0, BLOCK_SIZE_1) + i1
            mask1 = off1 < SIZE_1

            off_x = stride_x0 * off0[:, None] + stride_x1 * off1[None, :]
            off_y = stride_y0 * off0[:, None] + stride_y1 * off1[None, :]
            off_out = stride_o0 * off0[:, None] + stride_o1 * off1[None, :]

            mask = mask0[:, None] & mask1[None, :]

            x_val = tl.load(x_ptr + off_x, mask=mask)
            y_val = tl.load(y_ptr + off_y, mask=mask)
            res = x_val + y_val

            tl.store(out_ptr + off_out, res, mask=mask)


@torch._library.triton_op("testing::triton_add", mutates_args=())
def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    SIZE_0, SIZE_1 = x.size()
    out = torch.zeros((SIZE_0, SIZE_1), dtype=x.dtype, device=x.device)
    torch._library.capture_triton(addition_kernel)[(1,)](
        x,
        y,
        out,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_0,
        SIZE_1,
        64,
        16,
    )
    return out


def fn(x, y, z):
    r = x @ y
    return triton_add(r, z)


def get_input():
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((1024*16 - 7, 1024), device="cuda", dtype=torch.bfloat16).T
    z = torch.randn((1024, 1024*16 - 7), device="cuda", dtype=torch.bfloat16)
    return x, y, z

x, y, z = get_input()
expected = torch.compile(fn)(x, y, z)
actual = fn(x, y, z)
actual2 = fn(x, y, z)
torch.testing.assert_close(actual2, actual)
torch.testing.assert_close(expected, actual)