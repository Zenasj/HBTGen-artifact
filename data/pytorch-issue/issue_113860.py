import torch.nn as nn

import torch
import torch.fx
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch._export import capture_pre_autograd_graph
import random
import numpy as np
from transformers import pipeline
from diffusers import StableDiffusionPipeline

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

quant_model_name_list = []

quant_model_filter = ['vae', 'text_encoder', 'unet', 'safety_checker']
quant_model_filter = ['text_encoder']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WarppingModel(torch.nn.Module):
    def __init__(self, pipe, model, register_name):
        super(WarppingModel, self).__init__()
        self.pipe = pipe
        self.inner_model =  model
        self.register_name = register_name

    def forward(self, *args, **kwargs):
        res = self.inner_model(*args, **kwargs)
        gm = capture_pre_autograd_graph(self.inner_model, args, kwargs, None)
        print(self.inner_model.__class__,"success")

        return res
    
    def __getattr__(self, name):
      if name in self.__dict__.keys():
        return self.__dict__[name]
      if name in self._modules.keys():
        return self._modules[name]
      return getattr(self.inner_model,name)

def quantization_pipline(pipline):
    for k in pipline.__dict__.keys():
        if isinstance(pipline.__dict__[k], torch.nn.Module) and (k in quant_model_filter or len(quant_model_filter) == 0):
            quant_model_name_list.append(k)

    for k in quant_model_name_list:
        print("---------------------------------------",k)
        register_name = k
        if register_name == "vae":
            pipline.__dict__[k].decode = WarppingModel(pipline, pipline.__dict__[k].decode , register_name)
        else:
            pipline.__dict__[k] = WarppingModel(pipline, pipline.__dict__[k] , register_name)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")
quantization_pipline(pipe)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=8, width=8,num_inference_steps=1).images[0]  
print("over!")