py
import functools

import torch
import torch_tensorrt

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def generate_sd_unet_inputs():
    sample = torch.randn((2, 4, 64, 64), device="cuda", dtype=torch.float16)
    timestep = torch.rand([], device="cuda", dtype=torch.float32) * 999
    encoder_hidden_states = torch.randn((2, 77, 768), device="cuda", dtype=torch.float16)
    
    return sample, timestep, encoder_hidden_states

with torch.inference_mode():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    unet_model = pipe.unet.eval()
    unet_model.forward = functools.partial(unet_model.forward, return_dict=False)
    
    arg_inputs_unet = generate_sd_unet_inputs()
    expected_outputs_unet = unet_model(*arg_inputs_unet)
    
    unet_exported_program = torch.export.export(unet_model, arg_inputs_unet)
        
    with torch_tensorrt.logging.errors():
        compiled_unet = torch_tensorrt.dynamo.compile(
            unet_exported_program,
            inputs=arg_inputs_unet,
            enabled_precisions={torch.float16, torch.float32},
            truncate_double=True,
        )
    
    torch_tensorrt.save(compiled_unet, "sd_unet_compiled.ep", inputs=arg_inputs_unet)
    loaded_unet = torch.export.load("sd_unet_compiled.ep").module()