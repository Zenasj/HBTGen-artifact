import torch

torch.save(a_module_with_lstm_inside, path)  # with pytorch v1.7.1
torch.load(path).to(device)  # with pytorch v1.8.0