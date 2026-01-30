import torch
import torch.nn as nn

last_inner = inner_lateral #+ inner_top_down
results.insert(0, getattr(self, layer_block)(last_inner))

class uncover_model(torch.nn.Module):
    def __init__(self, fs):
        super(uncover_model, self).__init__()
        features = fs #fs is your full model
        bodylist = list(features.children())
        bodylist[-1] = bodylist[-1].head
        bodylist = bodylist[:-1] # backbone + fpn
        # bodylist[-1] = bodylist[-1][:-1] # uncomment to delete fpn
        self.features = torch.nn.ModuleList(bodylist)
        
    def forward(self, x, debug=False):
        for ii, fs in enumerate(self.features):
            x = fs(x)
        return x

checkmodel = model_custom(self.model)
checkmodel.eval()
with torch.no_grad():
     check_preds = checkmodel(image_list.tensors)
torch.onnx.export(checkmodel, image_list.tensors, "[PATH]model.onnx", output_names=output_names, opset_version=11, export_params=True)
#remember, the working one is opset_version 10, this one is to make it incorrect

img_preprocess = image_list.tensors.cpu().numpy()
np.save('input_pytorch.npy', img_preprocess)

for i in range(len(predictions)): 
    print(i, predictions[i].mean())

import onnxruntime as rt
import numpy as np
img_preprocess  = np.load('input_pytorch.npy', allow_pickle=True)
sess = rt.InferenceSession('[PATH]model.onnx')
input1 = sess.get_inputs()[0].name
outputs = sess.run(None, {input1: img_preprocess})