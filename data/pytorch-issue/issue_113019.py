import time

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer


# Create the Eager Model
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)

# Set the model to eval mode
model = model.eval()

# Create the data, using the dummy data here as an example
traced_bs = 1
x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
example_inputs = (x,)

# Capture the FX Graph to be quantized
with torch.no_grad():
    # if you are using the PyTorch nightlies or building from source with the pytorch master,
    # use the API of `capture_pre_autograd_graph`
    # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
    exported_model = capture_pre_autograd_graph(model, example_inputs)
    # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
    # please use the API of `torch._dynamo.export` to capture the FX Graph.
    # exported_model, guards = torch._dynamo.export(
    #     model,
    #     *copy.deepcopy(example_inputs),
    #     aten_graph=True,
    # )


optimized_model = torch.compile(model)
# Running some benchmark
iters = 100
count = 0
start_t = time.time()

res = optimized_model(*example_inputs)  # warmap

for i in range(iters):
    res = optimized_model(*example_inputs)
    count += 1

print("fp32 elapsed: ", time.time() - start_t, "count: ", count)

quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
prepared_model = prepare_pt2e(exported_model, quantizer)


prepared_model(*example_inputs)

converted_model = convert_pt2e(prepared_model)

optimized_model = torch.compile(converted_model)

# Running some benchmark
iters = 100
count = 0
start_t = time.time()

res = optimized_model(*example_inputs)  # warmap

for i in range(iters):
    res = optimized_model(*example_inputs)
    count += 1

print("int8 elapsed: ", time.time() - start_t, "count: ", count)

import time

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

print(torch._dynamo.list_backends())

# Create the Eager Model
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)

# Set the model to eval mode
model = model.eval()

# Create the data, using the dummy data here as an example
traced_bs = 1
x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
example_inputs = (x,)

# Capture the FX Graph to be quantized
with torch.no_grad():
    # if you are using the PyTorch nightlies or building from source with the pytorch master,
    # use the API of `capture_pre_autograd_graph`
    # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
    exported_model = capture_pre_autograd_graph(model, example_inputs)
    # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
    # please use the API of `torch._dynamo.export` to capture the FX Graph.
    # exported_model, guards = torch._dynamo.export(
    #     model,
    #     *copy.deepcopy(example_inputs),
    #     aten_graph=True,
    # )

with torch.no_grad():
    optimized_model = torch.compile(model)
# Running some benchmark
res = optimized_model(*example_inputs)  # warmap

iters = 100
count = 0
start_t = time.time()

with torch.no_grad():
    for i in range(iters):
        res = optimized_model(*example_inputs)
        count += 1

print("fp32 elapsed: ", time.time() - start_t, "count: ", count)

quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
prepared_model = prepare_pt2e(exported_model, quantizer)


prepared_model(*example_inputs)

converted_model = convert_pt2e(prepared_model)

with torch.no_grad():
    optimized_model = torch.compile(converted_model)

# Running some benchmark
res = optimized_model(*example_inputs)  # warmap

count = 0
start_t = time.time()

with torch.no_grad():
    for i in range(iters):
        res = optimized_model(*example_inputs)
        count += 1

print("int8 elapsed: ", time.time() - start_t, "count: ", count)