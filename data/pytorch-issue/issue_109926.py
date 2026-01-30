import torch
import importlib 
import sys

sys.path.append("benchmark")
c = "torchbenchmark.models.cm3leon_generate"
module = importlib.import_module(c)

device = "cuda"
benchmark_cls = getattr(module, "Model", None)
benchmark = benchmark_cls(test="eval", device = device)

model, example = benchmark.get_module()

eager = model(*example)

print("eager works")

print("be patient compilation takes a while on this model")
compiled = torch.compile(model)
compiled_out = compiled(*example)