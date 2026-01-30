import torch

dummy_input0 = torch.randn(1, 9, 1024, 768)
dummy_input1 = torch.randn(1, 7, 1024, 768)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    if not onnx_path.exists():
        torch.onnx.export(
            generator,
            (dummy_input0,
             dummy_input1),
            onnx_path,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")