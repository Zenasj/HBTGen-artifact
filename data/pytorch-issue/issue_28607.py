import torch
import torch.nn as nn

@torch.jit.script
def center_slice_helper(x, h_offset, w_offset, h_end, w_end):
    return x[:, :, h_offset:h_end, w_offset:w_end]


class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        Input shape can be dynamic
        :param crop_size: the center crop size
        """
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size
        self.register_buffer('crop_size_t', torch.tensor(crop_size))

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        """
        :type x: torch.Tensor
        """
        height, width = x.shape[2], x.shape[3]
        if not isinstance(height, torch.Tensor):
            height, width = torch.tensor(height).to(x.device), torch.tensor(width).to(x.device)
        h_offset = (height - self.crop_size_t) / 2
        w_offset = (width - self.crop_size_t) / 2
        h_end = h_offset + self.crop_size_t
        w_end = w_offset + self.crop_size_t
        return center_slice_helper(x, h_offset, w_offset, h_end, w_end)


model = CenterCrop(224)
onnxfile = "/mnt/output/gr/crop.onnx"

targets = ["cropped"]
dynamic_axes = {'data': [2, 3]}
dummy_input = torch.randn(1, 3, 300, 256, device='cpu').byte()
torch.onnx.export(model, dummy_input, onnxfile,
                  verbose=True, input_names=['data'],
                  dynamic_axes=dynamic_axes,
                  output_names=targets,
                  opset_version=10)

import onnxruntime as rt
sess = rt.InferenceSession(onnxfile)
outputs = sess.run(output_names, {'data': dummy_input.numpy()})