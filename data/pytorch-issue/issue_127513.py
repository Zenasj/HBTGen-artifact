import torch.nn as nn

py
# test.py
import argparse
import torch
import torch._inductor.config as config
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

def main(use_api_after_the_change):

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


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.conv(x))

    optimize_ctx = functools.partial(
        export_aot_inductor, device="cpu"
    )
    optimized_model_iter_fn = optimize_ctx
    example_inputs = (torch.randn(1, 3, 224, 224),)
    model = Model()
    model.eval()
    with torch.no_grad(), config.patch({"freezing": True}):
        for _ in range(3):
            optimized_model_iter_fn(model, example_inputs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--after", "-a", action="store_true", help="use the api after the regression"
    )
    args = parser.parse_args()
    main(args.after)
    print("done")