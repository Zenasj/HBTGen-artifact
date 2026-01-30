import torch

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx', strip_doc_string=False)

torch.onnx.export(MyModel(), (x,), 'test_export_arg.onnx', verbose=True)