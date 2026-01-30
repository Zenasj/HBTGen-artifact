import torch
import torchvision.models.quantization as models

def save_and_load(M):
    torch.save(M, 'b.pt')
    M.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    from torch.quantization import prepare, convert
    M = prepare(M)
    M = convert(M)
    print(M)
    torch.save(M, 'a.pt')
    a = torch.load('a.pt')
    a.eval()
    print(a)
    return

resnet18 = models.resnet18()
save_and_load(resnet18)