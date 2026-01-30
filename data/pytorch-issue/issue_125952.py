import torch.nn as nn

import logging
from dataclasses import dataclass
from datetime import datetime

import torch
import torch._inductor.config as inductor_config


logger = logging.getLogger(__name__)
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


@dataclass
class BenchmarkConfig:
    batch_size: int = 256
    enable_bf16: bool = True
    enable_pt2: bool = True
    device = "cuda:0"
    d_in = 2048


class SimpleModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim, bias=False)
        self.ts_encoding_params_dict = torch.nn.Parameter(
            torch.empty(
                [
                    2000,
                    dim,
                ]
            ).uniform_(-0.01, 0.01)
        )
        self.linear2 = torch.nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x,
        num_object,
        user_event_ts_buckets,
    ):
        emb = self.linear1(x)
        # user_event_ts_encoding = self.ts_encoding_params_dict[user_event_ts_buckets, :]
        user_event_ts_encoding = self.ts_encoding_params_dict.index_select(
            0, user_event_ts_buckets
        )
        emb = emb + user_event_ts_encoding
        res = self.linear2(emb)
        return res


def create_model_input(benchmark_config: BenchmarkConfig):
    batch_size = benchmark_config.batch_size
    d_in = benchmark_config.d_in
    device = benchmark_config.device

    dtype = torch.bfloat16 if benchmark_config.enable_bf16 else torch.float32

    x = torch.rand(
        batch_size * 1000, d_in, dtype=dtype, device=device
    ).requires_grad_()  # assuming seq_len_per_example is max_length // 2
    num_object = torch.tensor(
        [1000] * batch_size,
        dtype=torch.int,
        device=device,
    )

    user_event_ts_buckets = torch.randint(
        0,
        2000,
        (1000 * batch_size,),
        dtype=torch.int,
        device=device,
    )

    return (
        x,
        num_object,
        user_event_ts_buckets,
    )


def run_first_model_once(model, input):
    pred = model(*input)
    pred[0].sum().backward()


def single_run_benchmark():
    benchmark_config = BenchmarkConfig()
    model_input = create_model_input(benchmark_config)
    model = SimpleModel(benchmark_config.d_in)

    if benchmark_config.enable_bf16:
        model = model.to(dtype=torch.bfloat16)

    if benchmark_config.enable_pt2:
        inductor_config.decompose_mem_bound_mm = True
        inductor_config.trace.enabled = True
        model = torch.compile(model)
        model = model.to(benchmark_config.device)
        print("Start compiling model.")
        run_first_model_once(model, model_input)
    else:
        model = model.to(benchmark_config.device)

    # trace
    with torch.profiler.profile(with_flops=True) as profiler:
        for _ in range(5):
            run_first_model_once(model, model_input)

    trace_file_prefix = "{}".format(
        datetime.now().strftime(TIME_FORMAT_STR),
    )

    return


def main() -> None:
    single_run_benchmark()
    print("done")


if __name__ == "__main__":
    main()  # pragma: no cover