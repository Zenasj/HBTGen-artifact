import torch

def get_free():
    import subprocess
    r = subprocess.run(["free", "-m"], capture_output=True)
    d = r.stdout.decode('utf-8')
    s = d.split(':')[1].split()
    return f"[used={s[1]:7}, shared={s[3]:7}] "

pin=False
print(get_free() + f"Starting torch={torch.__version__} with pin_memory={pin}")

gpu = torch.rand(4347592704, dtype = torch.float16, device='cuda')
cpu = torch.empty(4347592704, dtype = torch.float16, device='cpu', pin_memory=pin)

print(get_free() + "Copying")
cpu.storage().copy_(gpu.storage(), non_blocking=False)
print(get_free() + "Copy finished")