import torch

py 
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
config = get_efficientdet_config('efficientdet_d1')                                                                                                               
net = EfficientDet(config, pretrained_backbone=True)                                                                                                                
net = DetBenchPredict(net)
net.eval()                                                                                                                                   
torch.onnx.export(net.cuda(),                                # model being run
                  (torch.randn(1, 3, 512, 512).cuda(), {"img_info": None}),    # model input (or a tuple for multiple inputs)
                  "effdet_all.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'],
                  opset_version=13, verbose=True)            # the model's output names