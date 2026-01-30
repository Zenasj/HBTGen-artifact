import torch

torch.onnx.export(model,
    (image, caption, cap_mask),
    "model.onnx",
    input_names=input_names,
    output_names=ouput_names,
    export_params=True
)