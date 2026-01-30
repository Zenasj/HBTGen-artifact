import onnx
import argparse
import torch
import torch.nn as nn

OPSET = 10
# OPSET = 11

class MinimalModel(nn.Module):
    def __init__(self):
        super(MinimalModel, self).__init__()
        self.constant_zero_pad = nn.ConstantPad2d((1, 0, 0, 0), 0)

    def forward(self, input_tensor):
        return self.constant_zero_pad(input_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('output_onnx')
    args = parser.parse_args()

    minimal_model = MinimalModel()
    minimal_model = nn.DataParallel(minimal_model)
    minimal_model.cuda()

    # Random deep feature
    input_tensor = torch.rand((1, 32, 128, 128))
    # Check model can do a forward pass
    minimal_model(input_tensor)
    # Export to onnx
    torch.onnx.export(
        minimal_model.module,
        (input_tensor),
        args.output_onnx,
        export_params=True, verbose=True, training=False, opset_version=OPSET
    )

    original_model = onnx.load(args.output_onnx)
    onnx.checker.check_model(original_model)