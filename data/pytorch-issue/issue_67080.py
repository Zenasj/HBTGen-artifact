import torch

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx', use_external_data_format=True)

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx')