import torch
import torchvision
import torch_tensorrt


model = torchvision.models.resnet18().eval().cuda()
model_jit = torch.jit.script(model)
# model_jit = torch.jit.trace(model, torch.rand((1, 3, 256, 256), device="cuda"))

trt_model = torch_tensorrt.ts.compile(
    model_jit,
    inputs=[torch_tensorrt.Input((1, 3, 256, 256))],
    device={
        "device_type": torch_tensorrt.DeviceType.GPU,
        "gpu_id": 0,
        "dla_core": 0,
        "allow_gpu_fallback": True,
    },
    enabled_precisions={torch.int8},
)

model = torchvision.models.resnet18().eval().cuda()

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 256, 256))],
    device={
        "device_type": torch_tensorrt.DeviceType.GPU,
        "gpu_id": 0,
        "dla_core": 0,
        "allow_gpu_fallback": True,
    },
    enabled_precisions={torch.int8},
)