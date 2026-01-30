import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn

logger = logging.getLogger(__name__)
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


@dataclass
class BenchmarkConfig:
    batch_size: int = 1024
    enable_bf16: bool = True
    enable_pt2: bool = False
    enable_unit_pt2_wrap: bool = True
    device = "cuda"
    dtype = torch.bfloat16
    d_model = 384
    top_k_value = 0.12
    max_length_dict = {"event_a": 1000, "event_b": 200, "event_c": 200}


class SimpleSequenceSummarizationBlock(nn.Module):

    # @lint-ignore FIXIT [TooManyArgsInFunction]
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        super().__init__()

        self.d_model = benchmark_config.d_model

        self.attn_norm = nn.LayerNorm(self.d_model)

        self.self_gating = nn.Linear(self.d_model, self.d_model)
        self.top_k_value = benchmark_config.top_k_value
        self.router = nn.Linear(self.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        bs, seq_len, d = x.shape
        # select top k tokens
        routing_w = self.router(x).squeeze(-1)  # shape [B, Seq]
        top_k_w, top_k_indices = torch.topk(
            routing_w, int(self.top_k_value * seq_len), dim=1
        )
        # top_k sorting
        top_k_indices_in_order, ordering = torch.sort(top_k_indices, dim=1)
        top_k_w_in_order = top_k_w.gather(dim=1, index=ordering)

        x_compute_indices = top_k_indices_in_order.unsqueeze(-1).expand(-1, -1, d)
        x_compute_scale = top_k_w_in_order.unsqueeze(-1).expand(-1, -1, d)
        x_compute = x.gather(dim=1, index=x_compute_indices)

        attn_output = self.attn_norm(x_compute)

        x_compute = x_compute + attn_output * x_compute_scale
        x_full = x.clone()
        x_full.scatter_(1, x_compute_indices, x_compute)
        x = x_full

        # Self gating
        x = self.self_gating(x)

        return x


class SimpleSeqSummarizationLayer(torch.nn.Module):
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        super().__init__()

        self.seq_summarization_modules: nn.ModuleDict = nn.ModuleDict()
        self.event_features_mapping = list(benchmark_config.max_length_dict.keys())
        for event_name in self.event_features_mapping:
            self.seq_summarization_modules[event_name] = (
                SimpleSequenceSummarizationBlock(
                    benchmark_config,
                )
            )

        self.enable_unit_pt2_wrap = benchmark_config.enable_unit_pt2_wrap
        if self.enable_unit_pt2_wrap:
            assert (
                not benchmark_config.enable_pt2
            ), "enable_unit_pt2_wrap and enable_pt2 cannot be both True"

    def forward(
        self,
        seq_embs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # seq_summarizations = []
        seq_embs_after_summarizations = {}
        for event_name in self.event_features_mapping:
            if self.enable_unit_pt2_wrap:
                curr_seq_embs_after_summarizations = torch.compile(
                    self.seq_summarization_modules[event_name],
                )(
                    seq_embs[event_name],
                )
            else:
                curr_seq_embs_after_summarizations = self.seq_summarization_modules[
                    event_name
                ](
                    seq_embs[event_name],
                )
            seq_embs_after_summarizations[event_name] = (
                curr_seq_embs_after_summarizations
            )

        return seq_embs_after_summarizations


def create_model_input(benchmark_config: BenchmarkConfig):
    batch_size = benchmark_config.batch_size
    device = benchmark_config.device

    dtype = benchmark_config.dtype

    x = {}
    for event_name in benchmark_config.max_length_dict.keys():
        x[event_name] = torch.randn(
            (
                batch_size,
                benchmark_config.max_length_dict[event_name],
                benchmark_config.d_model,
            ),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    return [x]


def run_first_model_once(model, input):
    pred = model(*input)
    pred = [i.sum() for i in pred.values()]
    pred = torch.stack(pred).sum()
    pred.sum().backward(retain_graph=True)


def single_run_benchmark():
    benchmark_config = BenchmarkConfig()
    model_input = create_model_input(benchmark_config)
    model = SimpleSeqSummarizationLayer(benchmark_config)

    model = model.to(benchmark_config.device)
    if benchmark_config.enable_bf16:
        model = model.to(dtype=torch.bfloat16)

    if benchmark_config.enable_pt2:
        inductor_config.decompose_mem_bound_mm = True
        inductor_config.trace.enabled = True
        model = torch.compile(model)
        model = model.to(benchmark_config.device)
        print("Start compiling model.")
        run_first_model_once(model, model_input)

    # trace
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=1),
        record_shapes=True,
        profile_memory=True,
        # with_stack=True,
    ) as profiler:
        for _ in range(10):
            run_first_model_once(model, model_input)
            profiler.step()

    return


def main() -> None:
    single_run_benchmark()
    print("done")


if __name__ == "__main__":
    main()