import torch

if __name__ == '__main__':
    model = mobilenet_v2(pretrained=False, progress=True, quantize=True)
    torch.jit.save(torch.jit.script(model), 'mov2.pt')