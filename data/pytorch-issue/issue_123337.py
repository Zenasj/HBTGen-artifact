import torch.nn as nn

import torch
# import pointnet_cls
from pointnet2_sem_seg_msg import get_model

import torch
import torch.nn.parallel
import torch.utils.data

# from data_utils.data_preprocess import compute_steps_num
###显示相关###
# from Lib.visualizelib import *
###最远点采样###
from pointnet2_utils import *
# import sem_autuomatic

point_num = 2048
class_num = 12
normal_channel = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model= get_model(class_num)
model=model.to(device)
model.eval()

checkpoint = torch.load('/home/skywalker/quardroped/src/Staircase/stair_recognize/pointnet2_pytorch_part_seg/models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

x = (torch.rand(1, 3, point_num) )
x = x.cuda() 

export_onnx_file = "best.onnx"
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version = 20)