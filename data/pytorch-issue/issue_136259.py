import torch
import numpy as np
import os


if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
    device = devices[-2]
    final_device = "cuda:" + device
else:
    final_device = 'cpu'


def loss_yolo_torch():
    from network.cv.yolov4.yolov4_pytorch import yolov4loss_torch
    return yolov4loss_torch()

y_true_0 = np.load('./output_torch[0][0].npy')
yolo_out1 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[0][1].npy')
yolo_out2 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[0][2].npy')
yolo_out3 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[1][0].npy')
yolo_out4 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[1][1].npy')
yolo_out5 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[1][2].npy')
yolo_out6 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[2][0].npy')
yolo_out7 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[2][1].npy')
yolo_out8 = torch.from_numpy(y_true_0).to(final_device)
y_true_0 = np.load('./output_torch[2][2].npy')
yolo_out9 = torch.from_numpy(y_true_0).to(final_device)

yolo_out = ((yolo_out1,yolo_out2,yolo_out3),(yolo_out4,yolo_out5,yolo_out6),(yolo_out7,yolo_out8,yolo_out9))


batch_y_true_0_torch = np.load('./batch_y_true_0_torch.npy')
batch_y_true_0_torch = torch.from_numpy(batch_y_true_0_torch).to(final_device)

batch_y_true_1_torch = np.load('./batch_y_true_1_torch.npy')
batch_y_true_1_torch = torch.from_numpy(batch_y_true_1_torch).to(final_device)

batch_y_true_2_torch = np.load('./batch_y_true_2_torch.npy')

batch_y_true_2_torch = torch.from_numpy(batch_y_true_2_torch).to(final_device)


batch_gt_box0_torch = np.load('./batch_gt_box0_torch.npy')
batch_gt_box0_torch = torch.from_numpy(batch_gt_box0_torch).to(final_device)

batch_gt_box1_torch = np.load('./batch_gt_box1_torch.npy')
batch_gt_box1_torch = torch.from_numpy(batch_gt_box1_torch).to(final_device)


batch_gt_box2_torch = np.load('./batch_gt_box2_torch.npy')
batch_gt_box2_torch = torch.from_numpy(batch_gt_box2_torch).to(final_device)

input_shape = np.load('./input_shape.npy')
input_shape = torch.from_numpy(input_shape).to(final_device)

loss_torch = loss_yolo_torch()


loss_torch_result = loss_torch(yolo_out, batch_y_true_0_torch, batch_y_true_1_torch,
                                    batch_y_true_2_torch, batch_gt_box0_torch, batch_gt_box1_torch,
                                    batch_gt_box2_torch, input_shape)

yolo_out1 = torch.isnan(yolo_out1).any()
print('yolo_out1;',yolo_out1) 
yolo_out2 = torch.isnan(yolo_out2).any()
print('yolo_out2;',yolo_out2)  
yolo_out3 = torch.isnan(yolo_out3).any()
print('yolo_out3;',yolo_out3) 
yolo_out4 = torch.isnan(yolo_out4).any()
print('yolo_out4;',yolo_out4) 
yolo_out5 = torch.isnan(yolo_out5).any()
print('yolo_out5;',yolo_out5)  
yolo_out6 = torch.isnan(yolo_out6).any()
print('yolo_out6;',yolo_out6)  
yolo_out7 = torch.isnan(yolo_out7).any()
print('yolo_out7;',yolo_out7) 
yolo_out8 = torch.isnan(yolo_out8).any()
print('yolo_out8;',yolo_out8)
yolo_out9 = torch.isnan(yolo_out9).any()
print('yolo_out9;',yolo_out9) 
batch_y_true_0_torch = torch.isnan(batch_y_true_0_torch).any()
print('batch_y_true_0_torch;',batch_y_true_0_torch) 
batch_y_true_1_torch = torch.isnan(batch_y_true_1_torch).any()
print('batch_y_true_1_torch;',batch_y_true_1_torch)  
batch_y_true_2_torch = torch.isnan(batch_y_true_2_torch).any()
print('batch_y_true_2_torch;',batch_y_true_2_torch) 
batch_gt_box0_torch = torch.isnan(batch_gt_box0_torch).any()
print('batch_gt_box0_torch;',batch_gt_box0_torch) 
batch_gt_box1_torch = torch.isnan(batch_gt_box1_torch).any()
print('batch_gt_box1_torch;',batch_gt_box1_torch)  
batch_gt_box2_torch = torch.isnan(batch_gt_box2_torch).any()
print('batch_gt_box2_torch;',batch_gt_box2_torch)  
input_shape = torch.isnan(input_shape).any()
print('input_shape;',input_shape)  

loss_torch_result = torch.isnan(loss_torch_result).any()
print('loss_torch_result;',loss_torch_result)