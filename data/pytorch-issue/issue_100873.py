import onnx
import torch
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            torch_dtype=torch.float16,
                                            revision="fp16",
                                            subfolder="unet")
unet.cuda()
# unet.float()

with torch.inference_mode(), torch.autocast("cuda"):
    inputs = torch.randn(2,4,64,64, dtype=torch.float16, device='cuda'), torch.randn(1, dtype=torch.float16, device='cuda'), torch.randn(2, 77, 768, dtype=torch.float16, device='cuda')

    # Export the model
    torch.onnx.export(unet,
                    inputs,
                    "unet.onnx",
                    opset_version=14,
                    do_constant_folding=False,  # whether to execute constant folding for optimization
                    # do_constant_folding=True, 
                    input_names = ['input_0', 'input_1', 'input_2'],
                    output_names = ['output_0'])