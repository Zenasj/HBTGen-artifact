import torch.nn as nn

from torch.autograd import Variable
import torch


class custom_net(torch.nn.Module):

    def __init__(self,):
        super(custom_net, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, martix, vector):
        martix = self.sigmoid(martix)
        vector = self.sigmoid(vector)
        outputs = torch.mv(martix, vector)
        return outputs


if __name__ == "__main__":

    net = custom_net()
    net = net.eval().cuda()
    martix = Variable(torch.randn(2, 3)).type(torch.FloatTensor)
    vector = Variable(torch.randn(3)).type(torch.FloatTensor)

    martix = martix.cuda()
    vector = vector.cuda()

    input_name = ["martix", "vector"]
    output_name = ["output"]

    torch.onnx.export(net, (martix, vector), "net.onnx", input_names=input_name, output_names=output_name,
                      opset_version=11)
    print("done!")