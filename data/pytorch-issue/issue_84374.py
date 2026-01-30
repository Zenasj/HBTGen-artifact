from turtle import xcor
import torch
import torch.nn as nn


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        out = nn.functional.interpolate(x, size=(x.shape[-2], x.shape[-1]))

        return out + x


if __name__ == '__main__':


    x = torch.randn(4, 64, 48, 48)

    model = TestModel()
    output = model(x)
    print(output.size())
    
    x_onnx = torch.randn(4, 64, 16, 25).numpy()
    onnx_pth = 'test.onnx'
    output_names = ['output']
    torch.onnx.export(
        model, x, onnx_pth,
        export_params=True,
        verbose=True,
        opset_version=11,  # 9 ~ 14
        input_names=['input'],
        output_names=output_names
    )

    import onnxruntime as rt
    opt = rt.SessionOptions()
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = rt.InferenceSession(onnx_pth, opt, providers=['CPUExecutionProvider'])
    onnx_output = sess.run(output_names, {'input': x_onnx})
    print(onnx_output.shape)