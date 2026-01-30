## first file is setpy.py
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension


setup(
    name='pytorch_loss',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'tmp_cpp',
            ['tmp.cu']),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages()
)

## second file is tmp.py which is the test case
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


##
# version 1: use torch.autograd
class LossObj1(nn.Module):
    def __init__(self):
        super(LossObj1, self).__init__()

    def forward(self, logits, label):
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float()
        label = label.view(-1)

        probs = logits.softmax(dim=0)

        errs_sort, _ = torch.sort(probs, dim=1, descending=True)

        losses = torch.einsum('ab,ab->a', errs_sort, probs)

        losses = losses.sum()
        return losses


##
# version 3: use cuda
import tmp_cpp
class LossObj3Func(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, ignore_index):
        losses, jacc = tmp_cpp.tmp_forward(logits,
                labels, ignore_index)
        ctx.vars = logits, labels, jacc, ignore_index
        return losses


class LossObj3(nn.Module):
    def __init__(self, ignore_index=-100):
        super(LossObj3, self).__init__()

    def forward(self, logits, label):
        losses = LossObj3Func.apply(logits, label, 255)
        losses = losses.sum()
        return losses


if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    import torchvision
    torch.manual_seed(15)
    torch.backends.cudnn.deterministic = True

    class Model(nn.Module):
        def __init__(self, n_classes):
            super(Model, self).__init__()
            net = torchvision.models.resnet18(pretrained=False)
            self.conv1 = net.conv1
            self.bn1 = net.bn1
            self.maxpool = net.maxpool
            self.relu = net.relu
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)
        def forward(self, x):
            feat = self.conv1(x)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feat = self.maxpool(feat)
            feat = self.layer1(feat)
            feat = self.layer2(feat)
            feat = self.layer3(feat)
            feat = self.layer4(feat)
            feat = self.fc(feat)
            out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
            return out

    c = 4
    net1 = Model(c)
    net2 = Model(c)
    net2.load_state_dict(net1.state_dict())
    criteria1 = LossObj1()
    criteria2 = LossObj3()
    net1.cuda()
    net2.cuda()
    net1.train()
    net2.train()
    criteria1.cuda()
    criteria2.cuda()

    bs, h, w = 2, 40, 40

    inten = torch.randn(bs, 3, h, w).cuda()
    lbs = torch.randint(0, c, (bs, h, w)).cuda()

    logits1 = net1(inten)
    loss1 = criteria1(logits1, lbs)
    logits2 = net2(inten)
    loss2 = criteria2(logits2, lbs).mul(weight).sum()