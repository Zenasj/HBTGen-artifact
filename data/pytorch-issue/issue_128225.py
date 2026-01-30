import torch
import faulthandler
faulthandler.enable()

neural_network_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                      pretrained=True)

model_scripted = torch.jit.script(neural_network_model.cpu())
model_scripted.save(f'test.pt')