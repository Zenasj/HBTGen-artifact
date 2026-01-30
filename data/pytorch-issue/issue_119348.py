import torch
import triton
import triton.language as tl


@triton.jit
def fwd_kernel(
    X_ptr,
    W1_ptr,
    b1_ptr,
    O_ptr,
    M: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C2: tl.constexpr,
):
    # Get program ids
    pid_m = tl.program_id(0)
    
    # Compute offsets
    offs_c1 = tl.arange(0, C1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Load input data
    x_block_ptr = X_ptr + offs_m[:, None] * C1 + offs_c1[None, :]
    x = tl.load(x_block_ptr)

    # Compute gating
    for c2 in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):

        # Compute block pointers
        offs_c2 = c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
        o_block_ptr = O_ptr + offs_m[:, None] * C2 + offs_c2[None, :]
        w1_block_ptr = W1_ptr + offs_c1[:, None] * C2 + offs_c2[None, :]
        b1_block_ptr = b1_ptr + offs_c2

        # Compute output
        w = tl.load(w1_block_ptr)
        b = tl.load(b1_block_ptr)
        o = tl.dot(x, w, allow_tf32=False)
        o += b[None, :]

        # Store output
        tl.store(o_block_ptr, o)


@triton.jit
def bwd_kernel(
    grad_out_ptr,
    grad_x_ptr,
    W1_ptr,
    M: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C2: tl.constexpr,
):
    # Get program ids
    pid_m = tl.program_id(0)

    # Compute offsets
    offs_c1 = tl.arange(0, C1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Compute gradient
    grad_x = tl.zeros((BLOCK_SIZE_M, C1), dtype=tl.float32)
    for c2 in range(0, tl.cdiv(C2, BLOCK_SIZE_C2)):

        # Compute block pointers
        offs_c2 = c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
        w1_block_ptr = W1_ptr + offs_c1[:, None] * C2 + offs_c2[None, :]
        go_block_ptr = grad_out_ptr + offs_m[:, None] * C2 + offs_c2[None, :]

        # Compute partial x gradient
        go = tl.load(go_block_ptr)
        w1 = tl.load(w1_block_ptr)
        w1 = tl.trans(w1)
        grad_x += tl.dot(go, w1, allow_tf32=False)

    # Store output
    gx_block_ptr = grad_x_ptr + offs_m[:, None] * C1 + offs_c1[None, :]
    tl.store(gx_block_ptr, grad_x)


def fwd(X, W1, b1):
    B, C1 = X.shape
    C1, C2 = W1.shape

    O = torch.empty(B, C2, dtype=X.dtype, device=X.device)

    BLOCK_SIZE_B = 64
    BLOCK_SIZE_C2 = 64
    grid = lambda META: (triton.cdiv(B, BLOCK_SIZE_B),)
    fwd_kernel[grid](
        X,
        W1,
        b1,
        O,
        B,
        C1,
        C2,
        BLOCK_SIZE_B,
        BLOCK_SIZE_C2,
    )
    return O


def bwd(grad_out, X, W1):
    B, C2 = grad_out.shape
    C1 = X.shape[-1]

    grad_x = torch.empty(B, C1, dtype=X.dtype, device=X.device)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_C2 = 64
    grid = lambda META: (triton.cdiv(B, BLOCK_SIZE_M),)
    bwd_kernel[grid](
        grad_out,
        grad_x,
        W1,
        B,
        C1,
        C2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_C2,
    )

    return grad_x, None, None


class test(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W1, b1):
        out = fwd(X, W1, b1)
        ctx.save_for_backward(X, W1)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, W1 = ctx.saved_tensors
        return bwd(grad_out, X, W1)


@torch.compile
def compiled(x, W1, b1):
    out = test.apply(x, W1, b1)
    return out


def test_fn(x, W1, b1):
    out = test.apply(x, W1, b1)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)


def compiled_test_fn(x, W1, b1):
    out = compiled(x, W1, b1)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)


def main():
    x = torch.randn(1024, 128, device='cuda', requires_grad=True)
    W1 = torch.randn(128, 256, device='cuda', requires_grad=False)
    b1 = torch.randn(256, device='cuda', requires_grad=False)

    # PASSES
    test_fn(x, W1, b1)

    # BREAKS (only sometimes?)
    compiled_test_fn(x, W1, b1)


if __name__ == "__main__":
    main()