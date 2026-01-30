py
import torch
import torch_tensorrt
import torchvision.models as models

model = models.resnet18().eval().cuda()
input = torch.randn((1, 3, 224, 224)).to("cuda")
compile_spec = {
        "inputs": [
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

exp_program = torch_tensorrt.dynamo.trace(model, **compile_spec)
trt_module = torch_tensorrt.dynamo.compile(exp_program, **compile_spec)
torch_tensorrt.save(trt_module, "./trt.ep", inputs=[input])
ep = torch_tensorrt.load("./trt.ep")