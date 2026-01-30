import numpy as np

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None)

unet_kwargs = {}
unet_kwargs["hidden_states"] = torch.ones((2, 16, 64, 64))
unet_kwargs["timestep"] = torch.from_numpy(np.array([1, 2], dtype=np.float32))
unet_kwargs["encoder_hidden_states"] = torch.ones((2, 154, 4096))
unet_kwargs["pooled_projections"] = torch.ones((2, 2048))

#Feature map height and width are dynamic
fm_height = torch.export.Dim('fm_height', min=16)
fm_width = torch.export.Dim('fm_width', min=16)

#iterate through the unet kwargs and set only hidden state kwarg to dynamic
dynamic_shapes = {key: (None if key != "hidden_states" else {2: fm_height, 3: fm_width}) for key in unet_kwargs.keys()}
transformer = torch.export.export_for_training(pipe.transformer.eval(), args=(), kwargs=(unet_kwargs), dynamic_shapes=dynamic_shapes).module()

#Feature map height and width are dynamic
fm_height = torch.export.Dim('fm_height', min=16, max=256)
fm_width = torch.export.Dim('fm_width', min=16, max=256)
dim = torch.export.Dim('dim', min=1, max=16)
fm_height = 16*dim
fm_width = 16*dim
dynamic_shapes = {"sample": {2: fm_height, 3: fm_width}}

fm_height = torch.export.Dim.DYNAMIC
fm_width = torch.export.Dim.DYNAMIC