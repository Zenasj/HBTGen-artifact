import torch

torch.onnx.export(
    model,                       # MyPyTorch Model
    features,                    # Dummy_Input tensor
    "./converted.onnx",            # Output file 
    opset_version=12,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['prediction'] # Output tensor name (arbitary)
)
...