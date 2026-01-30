import torch

cpu = torch.device("cpu")
gpu = torch.device("mps")

data = torch.rand((1, 2, 2, 3))

def run_test(device):
    global data
    with torch.no_grad():
        data = data.to(device)
        data = data.permute(0, 3, 1, 2)
        return data

cr = run_test(cpu)
gr = run_test(gpu)

print(cr)
print(gr)

print(cr.shape)
print(gr.shape)

print(cr.numpy())
print(gr.cpu().numpy())