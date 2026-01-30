import torch
import torch.nn as nn

dummy_input = torch.randn(config.BATCH_SIZE, 3, 96, 96).to(cuda_device)
torch.onnx.export(model, dummy_input, path + "lol.onnx", verbose=False)

if isinstance(model, torch.nn.DataParallel):
        model = model.module

torch.nn.DataParallel

model.module

if isinstance(model, torch.nn.DataParallel):
        raise ValueError('torch.nn.DataParallel is not supported by ONNX '
                         'exporter, please use \'attribute\' module to '
                         'unwrap model from torch.nn.DataParallel. Try '
                         'torch.onnx.export(model.module, ...)')

model = torch.nn.DataParallel(model)
onnx_model = torch.onnx.export(model.module, ...)