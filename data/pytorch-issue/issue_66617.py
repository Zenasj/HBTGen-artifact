import torch

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx', _retain_param_name=True)

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx')