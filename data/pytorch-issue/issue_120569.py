import copy
import torch
import torch.nn as nn
from typing import List, Dict

class BuggyNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(BuggyNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.RELU = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.bn1, self.RELU)

    def forward(self, x):
        out = x
        out = self.net(out)
        return out
    
class OkayNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(OkayNet, self).__init__()
        conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        bn1 = nn.BatchNorm2d(32)
        RELU = nn.ReLU()
        self.net = nn.Sequential(conv1, bn1, RELU)

    def forward(self, x):
        out = x
        out = self.net(out)
        return out

def test_load_state_dict(trained_weights: Dict[str, torch.Tensor], main: nn.Module):
    main.load_state_dict(trained_weights)
    main_new_state_dict = main.state_dict()

    # compare loaded items
    properly_loaded = True
    for k1, v1 in trained_weights.items():
        v2 = main_new_state_dict[k1]

        if v1.ndim == 0:
            if torch.norm(v1 - v2, p=1) > 1e-6:
                print(
                    f"Wrong weights for key {k1}, norm_diff = {torch.norm(v2-v1, p=1)}"
                )
                properly_loaded = False
        elif not torch.allclose(v1, v2):
            print(f"Wrong weights for key {k1}, norm_diff = {torch.norm(v2-v1, p=1)}")
            properly_loaded = False

    if properly_loaded:
        print("[+] Correct behavior: Loaded weights match the weights that were meant to be loaded.")
    else:
        print(
            "[-] Unexpected behavior: Loaded weights are different from the weights that were meant to be loaded."
        )

@torch.no_grad()
def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    return weights_avg


if __name__ == "__main__":
    num_clients = 2
    ### Bug experiment:
    buggy_main_model = BuggyNet()
    # average multiple clients
    client_models = [BuggyNet() for _ in range(num_clients)]
    buggy_avg_weights = average_weights([x.state_dict() for x in client_models])
    # Test if load_state_dict works properly
    test_load_state_dict(buggy_avg_weights, buggy_main_model)
    print("-" * 40)

    # ### Okay experiment:
    okay_main_model = OkayNet()
    client_models = [OkayNet() for _ in range(num_clients)]
    okay_avg_weights = average_weights([x.state_dict() for x in client_models])

    test_load_state_dict(okay_avg_weights, okay_main_model)
    print("-" * 40)