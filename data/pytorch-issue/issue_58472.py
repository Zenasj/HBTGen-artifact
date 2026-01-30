import torch

checkpoint = torch.load(model_path)
keypoint_net = KeypointNet(use_color=True, do_upsample=True, do_cross=True)
keypoint_net.load_state_dict(checkpoint['state_dict'])
keypoint_net = keypoint_net.cuda()
keypoint_net.eval()
dummy_input = torch.randn(1,3,224,338).cuda()
torch.onnx.export(keypoint_net,(dummy_input, ),'model.onnx', opset_version=11)