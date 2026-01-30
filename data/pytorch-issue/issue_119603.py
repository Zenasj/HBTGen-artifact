import json
import numpy as np
import skimage.io
import torch
import tqdm
import sys
import os
sys.path.append('./satlas')
import satlas.model.evaluate
import satlas.model.model
import os
# torch.set_float32_matmul_precision('high') causes error
os.environ["TORCH_LOGS"] = "+dynamic"

weights_path = "./extracted_files/satlas_explorer_datasets/models/solar_farm/best.pth"

# Locations of model config and weights, and the input image.
config_path = './satlas/configs/satlas_explorer_solar_farm.txt'

# Read config and initialize the model.

with open(config_path, 'r') as f:
    config = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for spec in config['Tasks']:
    if 'Task' not in spec:
        spec['Task'] = satlas.model.dataset.tasks[spec['Name']]
model = satlas.model.model.Model({
    'config': config['Model'],
    'channels': config['Channels'],
    'tasks': config['Tasks'],
})

device="cuda"
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_im_ts = torch.randn((9*4,256,256)).to(device)
x = torch.stack([test_im_ts], dim=0)
outputs, _ = model(x)
output_directory = "compiled_model_satlas_without_float32_matmul_precision"
os.makedirs(output_directory, exist_ok=True)
# model should be 4 image time series 9 bands each
batch_dim = torch.export.Dim("batch", min=1, max=32)
so_path = torch._export.aot_compile(
    f = model,
    args = (x, ),
    # Specify the first dimension of the input x as dynamic
    # dynamic_shapes={"x": {0: batch_dim}},
    # Specify the generated shared library path
    options={
        "aot_inductor.output_path": os.path.join(os.getcwd(), output_directory, "satlas_pt2.so"),
        "max_autotune": True,
    },
)