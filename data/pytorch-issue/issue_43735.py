import torch
import torch.nn as nn

class DummyLayerNoBug(nn.Module):
    def __init__(self):
        super(DummyLayerNoBug, self).__init__()
    
    def forward(self, x):
        y = torch.cat([x, x], -1)
        return y

class DummyLayerBug(nn.Module):
    def __init__(self):
        super(DummyLayerBug, self).__init__()
    
    def forward(self, x):
        y = torch.cat([x + 0., x], -1)
        return y

class DummyLayerBugFix(nn.Module):
    def __init__(self):
        super(DummyLayerBugFix, self).__init__()
    
    def forward(self, x):
        y = torch.cat([x + 0., x], x.dim() - 1)
        return y


def test_DummyLaer_jit_vs_nojit(dummy_layer, device):
    input_tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).to(device)
    target_tensor = torch.tensor([[[0.0000, 1.0000, 2.0000, 0.0000, 1.0000, 2.0000],
                                   [3.0000, 4.0000, 5.0000, 3.0000, 4.0000, 5.0000]]]).to(device)
    dummy_layer_jit = torch.jit.script(dummy_layer).to(device)
    dummy_layer = dummy_layer.to(device)

    y = dummy_layer(input_tensor)
    print("y: {}\n{}".format(y.shape, y))
    assert (y == target_tensor).all().item()
    y_jit = dummy_layer_jit(input_tensor)
    print("\ny_jit: {}\n{}".format(y_jit.shape, y_jit))
    assert (y_jit == target_tensor).all().item()


if __name__ == "__main__":

    print("\n\n=================== DummyLayerNoBug CPU ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerNoBug(), device=torch.device("cpu"))

    print("\n\n=================== DummyLayerBugFix CPU ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerBugFix(), device=torch.device("cpu"))

    print("\n\n=================== DummyLayerBug CPU ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerBug(), device=torch.device("cpu"))


    print("\n\n=================== DummyLayerNoBug CUDA ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerNoBug(), device=torch.device("cuda"))

    print("\n\n=================== DummyLayerBugFix CUDA ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerBugFix(), device=torch.device("cuda"))

    print("\n\n=================== DummyLayerBug CUDA ===================")
    test_DummyLaer_jit_vs_nojit(dummy_layer=DummyLayerBug(), device=torch.device("cuda"))