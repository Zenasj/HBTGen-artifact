import torch.nn as nn

import copy
import random
import torch
from transformers import HieraConfig, HieraModel, PretrainedConfig

config = HieraConfig(
    embed_dim=8,
    image_size=[64, 64],
    patch_stride=[4, 4],
    patch_size=[7, 7],
    patch_padding=[3, 3],
    masked_unit_size=[8, 8],
    mlp_ratio=1.0,
    num_channels=3,
    depths=[1, 1, 1, 1],
    num_heads=[1, 1, 1, 1],
    embed_dim_multiplier=2.0,
    hidden_act='gelu',
    decoder_hidden_size=2,
    decoder_depth=1,
    decoder_num_heads=1,
    initializer_range=0.02)

setattr(config, "initializer_range", 1e-10)  # NOTE THIS LINE !!!

model = HieraModel(config).to("xpu")
model.eval()

def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device="xpu").view(shape).contiguous()

inputs = {"pixel_values": floats_tensor([13, 3, 64, 64])}

model.config.use_cache = False

with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

torch.set_printoptions(threshold=100000)

def dump_tensor(module, input, output):
    for i in output:
        if isinstance(i, torch.Tensor):
            print(f"OUTPUT: is_nan={torch.isnan(i).any()}, isinf={torch.isinf(i).any()}")
    print("OUTPUT", module.__class__.__name__, output)

for name, module in model.named_modules():
    module.register_forward_hook(dump_tensor)

import torch
torch.set_printoptions(threshold=100000)

device='xpu'

input = torch.load("PT_141642.pt", map_location=device)
input=input[0]
print("Verifying input that there is no nan/inf:")
print("isnan:", torch.isnan(input).any())
print("isinf:", torch.isinf(input).any())

# (16,), eps and affine are taken from the Hiera model description for 4th LayerNorm operation
layer_norm = torch.nn.LayerNorm((16,), eps=1e-06, elementwise_affine=True, device=device)

with torch.no_grad():
    output = layer_norm(input)

print("Verifying output that there is no nan/inf:")
print("isnan:",  torch.isnan(output).any())
print("isinf:", torch.isinf(output).any())