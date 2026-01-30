import torch
import torch.nn as nn


class TestModel(nn.Module):

    def __init__(self, num_features, init_size=None):
        super(TestModel, self).__init__()

        if init_size is None:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(size=init_size * 2, mode='bilinear', align_corners=True)

        self.innorm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x):
        out = self.upsample(x)
        out = self.innorm(out)
        return out


if __name__ == '__main__':

    init_size = 64
    num_features = 32
    x = torch.randn(4, num_features, init_size, init_size)

    model = TestModel(num_features, init_size=None)
    # model = TestModel(num_features, init_size=init_size)
    output = model(x)
    print(output.size())
    # torch.Size([4, 32, 128, 128])

    torch.onnx.export(
        model, x, 'test.onnx',
        export_params=True,
        verbose=True,
        opset_version=14,  # 9 ~ 14
        input_names=['x'],
        output_names=['output']
    )