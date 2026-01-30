import torch

def save_jit():
    import torchvision
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    jit_model=torch.jit.script(model)
    torch.jit.save(jit_model, "jit.pk")


def load_save():
    loaded_model=torch.jit.load("jit.pk")
    torch.save({"ddo": loaded_model.state_dict()}, "foo.pth")

save_jit()
load_save()