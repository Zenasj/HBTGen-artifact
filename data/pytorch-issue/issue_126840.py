import os, shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from optimum.exporters.onnx.model_configs import FalconOnnxConfig

device = 'cuda'
model_name = "tiiuae/falcon-rw-1b" 
# model_name = "tiiuae/falcon-7b"

# load model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval().to(device)

# get data
model_config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float32)
onnx_config = FalconOnnxConfig(model_config)
data = onnx_config.generate_dummy_inputs()
for k, v in data.items():
    print(k)
    data[k] = v.to(device)

# create export directory
model_dir = model_name.replace('/', '_')
_, name = model_name.split('/')
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)

dynamo_export = False
if dynamo_export:
    export_output = torch.onnx.dynamo_export(
        model,
        **data
    )
    export_output.save(os.path.join(model_dir, 'HF_' + name + '_dynamo.onnx'))
else:
    torch.onnx.export(model, tuple(data.values()), os.path.join(model_dir, 'HF_' + name + '.onnx'))