import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)

        # upsample cause Gather error
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.expander(x)
        # a = self.P4_upsampled(x)  

        sh = torch.tensor(x.shape[-2:])
        print(sh)
        # a = F.interpolate(x, (sh[0]*2, sh[1]*2))
        a = F.interpolate(x, (544, 1920))
        return a


def export_onnx():
    model = TinyModel().to(device)
    sample_input = torch.rand(1, 3, 544, 1920).to(device)
    model.eval()
    torch.onnx.export(model, sample_input, model_p, input_names=[
                      'img'], output_names=['values'], opset_version=11)
    print('onnx model exported. forward now...')
    # forward now


if __name__ == "__main__":
    export_onnx()