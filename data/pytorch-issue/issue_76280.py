import torchvision

try:
  fasterrcnn_resnet50_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  print("Loaded fasterrcnn_resnet50_model!!!!!!")
except Exception as e:
  print("\n ###### \n Error Message while loading fasterrcnn_resnet50_model:\n",e,"\n ###### \n") 

try:
  fasterrcnn_mobilenet_v3_large_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  print("Loaded fasterrcnn_mobilenet_v3_large_model!!!!!!")
except Exception as e:
  print("\n ###### \n Error Message while loading fasterrcnn_mobilenet_v3_large_model:\n",e,"\n ###### \n") 

try:
  fasterrcnn_mobilenet_v3_large_320_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
  print("Loaded fasterrcnn_mobilenet_v3_large_320_model!!!!!!")
except Exception as e:
  print("\n ###### \n Error Message while loading fasterrcnn_mobilenet_v3_large_320_model:\n",e,"\n ###### \n") 

try:
  fcos_resnet50_model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
  print("Loaded fcos_resnet50_model!!!!!!")
except Exception as e:
  print("\n ###### \n Error Message while loading fcos_resnet50_model:\n",e,"\n ###### \n") 

try:
  retinanet_resnet50_model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
  print("Loaded retinanet_resnet50_model!!!!!!")
except Exception as e:
  print("Error Message while loading retinanet_resnet50_model:\n",e) 

try:
  ssd300_vgg16_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
  print("Loaded ssd300_vgg16!!!!!!")
except Exception as e:
  print("Error Message while loading ssd300_vgg16:\n",e) 

try:
  ssdlite320_mobilenet_v3_large_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
  print("Loaded ssdlite320_mobilenet_v3_large!!!!!!")
except Exception as e:
  print("Error Message while loading ssdlite320_mobilenet_v3_large:\n",e) 

try:
  maskrcnn_resnet50_fpn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
  print("Loaded maskrcnn_resnet50_fpn!!!!!!")
except Exception as e:
  print("Error Message while loading maskrcnn_resnet50_fpn:\n",e)