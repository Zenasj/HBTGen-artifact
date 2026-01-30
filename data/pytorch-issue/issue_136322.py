import torch
import os
import tempfile
import torch_tensorrt as torchtrt
import torchvision.models as models

torch.manual_seed(0)
trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")

model = models.resnet18().eval().cuda()
input = torch.rand((1, 3, 224, 224)).to("cuda")
inputs = [input]

exp_program = torch.export.export(model, tuple(inputs), strict=False)
# use torchtrt compiled model
compile_spec = {
    "inputs": [
        torchtrt.Input(
            input.shape, dtype=torch.float, format=torch.contiguous_format
        )
    ],
    "ir": "dynamo",
    "min_block_size": 1,
    "cache_built_engines": False,
    "reuse_cached_engines": False,
}
trt_model = torchtrt.dynamo.compile(exp_program, **compile_spec)
torchtrt.save(trt_model, trt_ep_path, inputs=inputs)
print(f"lan added torchtrt model saved in {trt_ep_path}")