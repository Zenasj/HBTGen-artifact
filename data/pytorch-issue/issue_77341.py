import torch

torch.onnx.export(
    model,
    (x,),
    "model.onnx",
    input_names=["input", "input2"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "input2": {0: "batch"},
        "output": {0: "batch"},
    },
)