py
import torch
import accelerate
from diffusers import UNet2DConditionModel

torch.manual_seed(0)
cuda_device = torch.device("cuda:0")

accelerator = accelerate.Accelerator(mixed_precision="fp16", dynamo_backend="inductor")

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", revision=None
)

unet.train()
unet.to(cuda_device)
unet = accelerator.prepare_model(unet)

x = torch.rand([2, 4, 64, 64], dtype=torch.float16, device=cuda_device)
t = torch.randint(0, 1000, size=(2,), device=cuda_device)
hidden = torch.rand([2, 77, 768], dtype=torch.float16, device=cuda_device)

model_pred = unet(x, t, hidden).sample
print(model_pred)

py
model = dynamo.optimize(self.state.dynamo_backend.value.lower())(model)
model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
model.forward = convert_outputs_to_fp32(model.forward)

model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
model.forward = convert_outputs_to_fp32(model.forward)
model = dynamo.optimize(self.state.dynamo_backend.value.lower())(model)

model = dynamo.optimize(self.state.dynamo_backend.value.lower())(model)
model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
model.forward = convert_outputs_to_fp32(model.forward)