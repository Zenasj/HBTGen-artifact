import torch.nn as nn

from pathlib import Path
import torch
from torch import nn
import kornia
import onnx
from onnxsim import simplify
import blobconverter

# Name of the model
name = 'threshold'

# Define the model class
class Model(nn.Module):

    def forward(self, image):
        kernel_size = (22, 22)
        kernel = torch.ones(kernel_size)
        blob = kornia.morphology.opening(image, kernel)
        return blob


# Define the expected input shape (dummy input)
shape = (1, 3, 300, 300)  # Example input shape
model = Model()
X = torch.ones(shape, dtype=torch.float32)

# Create output directory if it doesn't exist
path = Path("out/")
path.mkdir(parents=True, exist_ok=True)

# Export the model to ONNX format
onnx_path = str(path / (name + '.onnx'))
print(f"Exporting model to {onnx_path}")
torch.onnx.export(
    model,
    X,
    onnx_path,
    opset_version=12,
    do_constant_folding=True,
)

# Simplify the ONNX model
onnx_simplified_path = str(path / (name + '_simplified.onnx'))
print(f"Simplifying model to {onnx_simplified_path}")
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, onnx_simplified_path)

# Convert ONNX model to blob format
blobconverter.from_onnx(
    model=onnx_simplified_path,
    data_type="FP16",
    shaves=6,
    use_cache=False,
    output_dir="../models",
    optimizer_params=[]
)