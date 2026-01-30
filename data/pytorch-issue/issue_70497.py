torch.onnx.export(m,
              x,
              fn,
              export_params=True,
              opset_version=11,
              do_constant_folding=True,
              input_names=['input'],
              output_names=['output'],
              dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}})

import torch as t

from src_to_implement.model import ResNet

model = ResNet()

if __name__ == '__main__':
    x = t.randn(1, 3, 300, 300, requires_grad=True)
    m = model.cpu()
    m.eval()
    t.onnx.export(m,
                  x,
                  "onnx_export",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})