from unet import unet
import torch.onnx
import torch
from onnx_coreml import convert
import coremltools
import onnx




net = unet().cuda()
net.load_state_dict(torch.load('/content/drive/My Drive/Collab/f5/CelebAMask-HQ/face_parsing/model.pth'))


dummy = torch.rand(1,3,512,512).cuda()


torch.onnx.export(net, dummy, "Model.onnx", input_names=["image"], output_names=["output"])

onnx_model = onnx.load("Model.onnx")
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

finalModel = convert('Model.onnx', minimum_ios_deployment_target='13', image_output_names=["output"], image_input_names=["image"])

finalModel.save('ModelML.mlmodel')

torch.load('/content/drive/My Drive/Collab/f5/CelebAMask-HQ/face_parsing/model.pth', map_location=torch.device('cpu'))