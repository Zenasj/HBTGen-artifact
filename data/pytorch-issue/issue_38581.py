import torch.nn as nn
import torch

from torch.utils.data import DataLoader, RandomSampler

# Hook class
class ActiveGradsHook:
    def __init__(self, name):
        self.name = name # Runnable on 1.5

    def __call__(self, grad):
        try:
            return torch.zeros(grad.shape)
        except Exception as e:
            print(e)

def train_new_neurons(model):
    # Generate hooks for each layer
    for name, param in model.named_parameters():
        hook = ActiveGradsHook(name)
        param.register_hook(hook)

    # Train simply
    train(model)


def train(model):
    initial_weights = model.weight.clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0, weight_decay=1)
    
    inputs = torch.rand(2, 3)
    action_target = torch.rand(2, 2)

    action_output = model(inputs)
    
    loss = nn.MSELoss()(action_target, action_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(model.weight)
    print(initial_weights)

    assert initial_weights[0][0] == model.weight[0][0] # This fails

if __name__ == "__main__":
    model = nn.Linear(3, 2)
    train_new_neurons(model)