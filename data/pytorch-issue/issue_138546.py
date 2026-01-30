import torch
import triton
import triton.language as tl


def pytorch_foo(x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    if y is None:
        return x**2
    else:
        return (x + y) ** 2


@triton.jit
def _triton_foo(x_ptr, y_ptr, o_ptr, n):
    i = tl.program_id(0)
    x = tl.load(x_ptr + i, mask=i < n)
    if y_ptr is not None:
        y = tl.load(y_ptr + i, mask=i < n)
        x += y

        # Comment out this static_assert makes the compilation pass,
        # however, that would cause an illegal memory access.
        tl.static_assert(y.dtype == x.dtype)

    o = x * x
    tl.store(o_ptr + i, o, mask=i < n)


def triton_foo(x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    n = x.size(0)
    o = torch.empty_like(x)
    _triton_foo[(n,)](x, y, o, n)
    return o


@triton.jit
def _triton_bar(x_ptr, y_ptr, o_ptr, n, has_y_ptr: tl.constexpr):
    i = tl.program_id(0)
    x = tl.load(x_ptr + i, mask=i < n)
    if has_y_ptr:  # workaround for optional args
        y = tl.load(y_ptr + i, mask=i < n)
        x += y
        tl.static_assert(y.dtype == x.dtype)
    o = x * x
    tl.store(o_ptr + i, o, mask=i < n)


def triton_bar(x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    n = x.size(0)
    o = torch.empty_like(x)
    _triton_bar[(n,)](x, y, o, n, has_y_ptr=y is not None)
    return o


def test_optional_args() -> None:
    n = 20
    x = torch.rand((n,), dtype=torch.float16, device="cuda")
    compiled_foo = torch.compile(triton_foo, fullgraph=True, mode="max-autotune-no-cudagraphs")
    compiled_bar = torch.compile(triton_bar, fullgraph=True, mode="max-autotune-no-cudagraphs")

    y = torch.rand_like(x)
    o = pytorch_foo(x, y)
    torch.testing.assert_close(triton_foo(x, y), o)
    torch.testing.assert_close(compiled_bar(x, y), o)
    torch.testing.assert_close(compiled_foo(x, y), o)

    y = None
    o = pytorch_foo(x, y)
    torch.testing.assert_close(triton_foo(x, y), o)
    torch.testing.assert_close(compiled_bar(x, y), o)  # Workaround
    torch.testing.assert_close(compiled_foo(x, y), o)  # Problematic


if __name__ == "__main__":
    test_optional_args()