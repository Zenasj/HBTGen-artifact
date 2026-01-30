import torch

with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
    torch.onnx.export(
        pipeline, 
        (audio.cuda(),),
        "pipeline.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14
    )