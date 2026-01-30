import numpy as np
import cv2
import os
import torch
import torchvision.models as models

current_path = os.path.dirname(os.path.abspath(__file__))
testset_folder = os.path.join(current_path, '../datasets/widerface/validation/images')
testset_list = os.path.join(current_path, '../datasets/widerface/validation/wider_val.txt')

with open(testset_list, 'r') as fr:
    test_dataset = fr.read().split()

torch.set_grad_enabled(False)
model = models.resnet50()
model.eval()

device = torch.device('cpu')
model = model.to(device)

for i, filename in enumerate(test_dataset):
    print('{}/{}'.format(i, len(test_dataset)))
    img_path = os.path.join(testset_folder, filename)
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    model(img)