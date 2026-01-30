import torch

import importlib
from os.path import exists
import sys

from functorch._src.aot_autograd import aot_module
from functorch._src.compilers import nop

torchbench_dir = '../torchbenchmark'
assert exists(torchbench_dir), "../../torchbenchmark does not exist"

sys.path.append(torchbench_dir)
model_name = 'resnet50_quantized_qat'
module = importlib.import_module(f"torchbenchmark.models.{model_name}")
benchmark_cls = getattr(module, "Model", None)
if not hasattr(benchmark_cls, "name"):
    benchmark_cls.name = model_name

benchmark = benchmark_cls(
    test="train", device='cpu', jit=False, batch_size=2
)

model, example_inputs = benchmark.get_module()

fn = aot_module(model, nop)
out = fn(*example_inputs)