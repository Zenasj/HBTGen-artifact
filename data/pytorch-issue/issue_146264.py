import torch
import triton


def main():
    dtype = torch.float32
    dim = 1305301
    a = torch.rand(100, device="cuda", dtype=dtype)
    index = torch.randint(0, 100, (dim,), device="cuda")
    src = torch.rand(dim, device="cuda", dtype=dtype)

    print("=" * 20)
    print(
        triton.testing.do_bench(
            lambda: a.scatter_add(0, index, src),
            return_mode="median",
        )
    )
    print("=" * 20)

if __name__ == "__main__":
    main()