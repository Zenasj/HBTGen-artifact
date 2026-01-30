import torch.nn as nn

# pyre-strict
from typing import List, Optional, Tuple

import click
import pandas as pd

import torch

# @manual=//triton:triton
import triton


# CUDA_VISIBLE_DEVICEs=7 buck2 run @mode/opt //scripts/zhaozhu:cat_bench


@click.command()
@click.option("--data-type", type=str, default="bf16")
@click.option("--return-result", type=bool, default=False)
def main(
    data_type: str,
    return_result: bool,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if data_type == "fp32":
        dtype = torch.float32
    elif data_type == "fp16":
        dtype = torch.float16
    elif data_type == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {data_type}.")

    D1 = int(torch.randint(low=10000, high=50000, size=(1,)).item())
    D2 = int(torch.randint(low=100, high=1000, size=(1,)).item())
    D3 = int(torch.randint(low=500, high=1000, size=(1,)).item())

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["B"],
            x_vals=[100, 1000, 10000, 20000],
            line_arg="provider",
            line_vals=["pt_eager", "copy"],
            line_names=["pt_eager", "copy"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="ms",
            plot_name=f"torch-cat-D1-{D1}-D2-{D2}-D3-{D3}-dtype-{dtype}",
            args={
                "D1": D1,
                "D2": D2,
                "D3": D3,
                "dtype": dtype,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_cat(
        B: int,
        D1: int,
        D2: int,
        D3: int,
        dtype: torch.dtype,
        provider: str,
    ) -> float:
        warmup = 10
        rep = 3

        tensors = []

        a = torch.empty(
            # (B, 30108),
            (B, D1),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-1.0, 1.0)
        b = torch.empty(
            # (B, 624),
            (B, D2),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-1.0, 1.0)
        c = torch.empty(
            # (B, 772),
            (B, D3),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-1.0, 1.0)

        tensors = [a, b, c]

        total_cols: int = int(a.shape[1] + b.shape[1] + c.shape[1])

        def torch_copy(
            tensors: List[torch.Tensor], is_inplace: bool = True
        ) -> torch.Tensor:
            f = torch.zeros([B, total_cols], dtype=dtype, device=torch.device("cuda"))
            col_idx = 0
            for t in tensors:
                temp = f[:, col_idx : col_idx + t.shape[1]]
                if is_inplace:
                    temp.copy_(t)
                else:
                    f[:, col_idx : col_idx + t.shape[1]] = t
                col_idx += t.shape[1]
            return f

        def torch_cat(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat(tensors, dim=1)

        ref = torch_cat(tensors)
        real = torch_copy(tensors, is_inplace=False)

        torch.testing.assert_allclose(ref, real)

        if provider == "pt_eager":
            fn = lambda: torch_cat(tensors)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        elif provider == "stack":

            def torch_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
                return torch.stack(tensors, dim=1).view(-1, total_cols)

            fn = lambda: torch_stack(tensors)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        elif provider == "copy":
            fn = lambda: torch_copy(tensors)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        else:
            raise ValueError(f"unsupported provider: {provider}")

    df = bench_cat.run(print_data=True, return_df=return_result)

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()