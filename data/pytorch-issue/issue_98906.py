from torchvision import models
import torch
import copy

def make_contiguous(module):
    with torch.no_grad():
        state_dict = module.state_dict()
        state_dict=copy.deepcopy(state_dict)
        print("Non contiguous state dict:")
        for name, param in state_dict.items():
            if not param.is_contiguous():
                print(name)
                print(param.is_contiguous())
                state_dict[name] = param.contiguous()
                print(state_dict[name].is_contiguous())

    print("New non contiguous state dict:")
    module.load_state_dict(state_dict, assign=True)
    for name, param in module.state_dict().items():
        if not param.is_contiguous():
            print(name)
            print(param.is_contiguous())

def train(learning_rate):
    # Set up standard model.
    model = getattr(models, "resnet50")()
    model = model.to(memory_format=torch.channels_last)
    make_contiguous(model)

if __name__ == '__main__':
    train(0.1)