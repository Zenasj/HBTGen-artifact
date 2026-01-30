import torch.onnx
import torch.nn as nn
import onnxruntime as rt
import torch.nn.functional as F


class TraceCheck(nn.Module):
    def __init__(self):
        super(TraceCheck, self).__init__()

    def forward(self, x):
        return F.softmax(-100*x, -2)


if __name__ == '__main__':
    net = TraceCheck()
    x = torch.Tensor(torch.randn(5,5))
    pred = net.forward(x).detach().cpu().numpy()
    tmp_filename = 'test.onnx'
    torch.onnx.export(net, x, tmp_filename)
    sess = rt.InferenceSession(tmp_filename)
    pred_onnx = sess.run([sess.get_outputs()[-1].name], {sess.get_inputs()[-1].name: x.detach().cpu().numpy()})[0]
    print(pred)
    print(pred_onnx)