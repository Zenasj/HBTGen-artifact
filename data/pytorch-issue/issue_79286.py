import torch

x = torch.tensor([1, 2, 3, 4, 5])

types = torch.float, torch.int
device_names = "cpu", "mps"

for name in device_names:
    
    print(name)
    
    x = x.to(torch.device(name))
    
    print("x: ", x)
    print(x.type(), "\n")

    for t in types:
        print(f"x.to({t}): {x.to(t)}")
        
    print()
    
    mask = x > 2
    
    print("mask (x > 2): ", mask)

    for t in torch.float, torch.int:
        print(f"mask.to({t}): {mask.to(t)}")
        
    print("\n\n")