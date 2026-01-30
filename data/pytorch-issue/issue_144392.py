def to_numpy(tensor):
    return tensor.detach().gpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_batch)}
ort_outs = ort_session.run(None, ort_inputs)

print("Pytorch CUDA:", torch.cuda.is_available())
print("Available Providers:", onnxruntime.get_available_providers())
print("Active Providers for this session:", ort_session.get_providers())

import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
conv = conv.cuda()

input_tensor = torch.randn(1, 3, 224, 224).cuda()

try:
    output = conv(input_tensor)
    print("PyTorch cuDNN Test Passed!")
except Exception as e:
    print(f"PyTorch cuDNN Test Failed: {e}")