import torch.nn as nn

py
# test.py
import math
from typing import List, Tuple
import argparse
import torch
import torch._inductor.config as config
import torch.nn.functional as F

from typing import (
    Any,
    Mapping,
    Tuple,
)
import copy
import dataclasses
import functools
import weakref
from torch.utils import _pytree as pytree

def main(use_api_after_the_change, compile):

    def _normalize_bench_inputs(example_inputs) -> Tuple[Tuple[Any], Mapping[str, Any]]:
        # NOTE(bowbao): For huggingface benchmark, example_inputs are formatted as dictionary,
        # and consumed like `model(**example_inputs)`.
        # For other benchmarks, example_inputs are formatted as tuple and consumed
        # like `model(*example_inputs)`.
        if isinstance(example_inputs, dict):
            return (), example_inputs
        else:
            return tuple(example_inputs), {}

    def _register_dataclass_output_as_pytree(example_outputs) -> None:
        # NOTE(angelayi): For huggingface benchmark, some example outputs are
        # formatted as a dataclass which pytree cannot consume. So we want
        # to register the pytree implementation here
        example_outputs_flat = pytree.tree_leaves(example_outputs)
        output_dataclass_types = [
            type(out) for out in example_outputs_flat if dataclasses.is_dataclass(type(out))
        ]
        for output_type in output_dataclass_types:
            from torch._export.utils import register_dataclass_as_pytree_node

            register_dataclass_as_pytree_node(
                output_type,
                serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
            )

    class AOTInductorModelCache:
        cache = dict()

        @classmethod
        def load(cls, model, example_inputs, device):
            import torch._inductor
            import torch.export._trace

            key = weakref.ref(model)
            if key not in cls.cache:
                # Register the output dataclass to pytree
                example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
                with torch.no_grad():
                    # copy.deepcopy is required to prevent any surprising side-effect,
                    # see https://github.com/pytorch/pytorch/issues/113029
                    example_outputs = copy.deepcopy(model)(*example_args, **example_kwargs)

                if pytree._is_namedtuple_instance(example_outputs):
                    typ = type(example_outputs)
                    pytree._register_namedtuple(
                        typ,
                        serialized_type_name=f"{typ.__module__}.{typ.__name__}",
                    )
                else:
                    _register_dataclass_output_as_pytree(example_outputs)

                
                if use_api_after_the_change:
                    gm = torch.export._trace._export(
                        model,
                        example_args,
                        example_kwargs,
                        pre_dispatch=True,
                    ).module()
                else:
                    gm = torch.export._trace._export_to_torch_ir(
                        model,
                        example_args,
                        example_kwargs,
                    )
                
                with torch.no_grad():
                    so_path = torch._inductor.aot_compile(
                        gm, example_args, example_kwargs
                    )  # type: ignore[arg-type]

                cls.cache[key] = torch._export.aot_load(so_path, device)

            return cls.cache[key]

    def export_aot_inductor(model, example_inputs, device):
        optimized = AOTInductorModelCache.load(model, example_inputs, device)

        def opt_aot_inductor(_, example_inputs, collect_outputs=False):
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            return optimized(*example_args, **example_kwargs)

        return opt_aot_inductor

    # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
        if isinstance(x, torch.Tensor):
            return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
        else:
            return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

    # Dynamically pad input x with 'SAME' padding for conv with specified args
    def pad_same(
            x,
            kernel_size: List[int],
            stride: List[int],
            dilation: List[int] = (1, 1),
            value: float = 0,
    ):
        ih, iw = x.size()[-2:]
        pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
        pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
        return x


    class ReproModel(torch.nn.Conv2d):
        def __init__(
                self, in_channels, out_channels, kernel_size, stride=1, padding='SAME',
                dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
            padding, is_dynamic = 0, True
            super().__init__(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.gain = torch.nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
            self.scale = gamma * self.weight[0].numel() ** -0.5
            self.same_pad = is_dynamic
            self.eps = eps

        def forward(self, x):
            if self.same_pad:
                x = pad_same(x, self.kernel_size, self.stride, self.dilation)
            weight = F.batch_norm(
                self.weight.reshape(1, self.out_channels, -1), None, None,
                weight=(self.gain * self.scale).view(-1),
                training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
            return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


    config.profiler_mark_wrapper_call = True
    config.cpp.enable_kernel_profile = True

    if compile:
        optimize_ctx = functools.partial(
            torch.compile,
            backend="inductor",
        )
        optimized_model_iter_fn = optimize_ctx(lambda mod, inputs: mod(*inputs))
    else:
        optimize_ctx = functools.partial(
            export_aot_inductor, device="cpu"
        )
        optimized_model_iter_fn = optimize_ctx
    
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 2
    bias = True
    eps = 1e-05
    
    example_inputs = (torch.randn(128, in_channels, 192, 192),)
    model = ReproModel(in_channels, out_channels, kernel_size, stride, bias=bias, eps=eps)
    model.eval()
    with torch.no_grad(), config.patch({"freezing": True}):
        for _ in range(3):
            optimized_model_iter_fn(model, example_inputs)
        
        with torch.autograd.profiler.profile(True) as prof:
            optimized_model_iter_fn(model, example_inputs)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--after", "-a", action="store_true", help="use the api after the regression"
    )
    parser.add_argument(
        "--compile", "-c", action="store_true", help="use the api after the regression"
    )    
    args = parser.parse_args()
    
    with torch.backends.mkldnn.flags(enabled=False):
        main(args.after, args.compile)
    print("done")