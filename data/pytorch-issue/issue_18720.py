import torch
import torchvision

from dataloader import Mscoco

from SPPE.src.main_fast_inference import *
from torch.autograd import Variable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def LoadModel():
    pose_dataset = Mscoco()
    model = InferenNet_fast(4 * 1 + 1, pose_dataset)

    model = model.cuda()

    return model

if __name__ == '__main__':

    
    model = LoadModel()
    # print("model", model)

    x = Variable(torch.randn(1, 3, 320, 256).cuda(), requires_grad=True)
    result = model(x)
    print("torch result:", result)
    traced_script_module = torch.jit.trace(model, x)
    output = traced_script_module(x)
    print("--------------------------------")
    print("trave result:", output)
    traced_script_module.save("/data4/zjf/code/Pose-Estimation/AlphaPose-pytorch-v2/models/duc_se.pt")