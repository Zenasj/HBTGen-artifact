import torch.nn.functional as F
import numpy as np

def get_s_predictions(model, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x)
        y_prob = F.softmax(y_pred, dim=-1)
        top_pred = y_prob.argmax(1, keepdim=True)
        print("[Pytorch] prob list: ", y_prob)

    return top_pred

def infer_with_pytorch(img_path, pytorch_model):
    # Load the models:
    model = ResNet(config.resnet50_config, config.OUTPUT_DIM)
    model.load_state_dict(torch.load(pytorch_model))

    test_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds)
    ])

    img = Image.open(img_path)
    img_t = test_transforms(img)
    img_tensor = torch.unsqueeze(img_t, 0)
    vtype_class = str(classes[get_s_predictions(model, img_tensor)])
    return vtype_class

def infer_with_onnx(img_path, onnx_model):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    img = Image.open(img_path)
    resize = transforms.Resize([224, 224])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    normalize = transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
    img = normalize(img)
    img.unsqueeze_(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    out_y = ort_outs[0]
    print("[ONNX] onnx prob: ", out_y)
    e_x = np.exp(out_y - np.max(out_y))
    new_list = e_x / e_x.sum()  # only difference
    print("[ONNX] new list: ", new_list)
    detected_class = str(classes[out_y.argmax(axis=1)[0]])
    print("[ONNX] predicted class: ", detected_class)
    return detected_class

import torch
import onnx
from model import ResNet
import config

# define model and load weights 
model = ResNet(config.resnet50_config, config.OUTPUT_DIM)
model.load_state_dict(torch.load('Models/resnet-model-erasing.pt'))

device = 'cpu'
model = model.to(device)

# export model to ONNX
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))

input_names = ["data"]
output_names = ["output"]
torch.onnx.export(model,
                  dummy_input.to(device),
                  'new_onnx/resnet-erasing.onnx',
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=11)

# check created model
onnx_model = onnx.load('new_onnx/resnet-erasing.onnx')
onnx.checker.check_model(onnx_model)