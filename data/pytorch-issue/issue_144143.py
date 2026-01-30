import torch
is_xpu_available = torch.xpu.is_available()
print(f"xpu available: {is_xpu_available}")

if is_xpu_available:
    # 创建一些在GPU上运行的数据
    device = torch.device("xpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)

    # 将数据移动到GPU上
    x = x.to(device)
    y = y.to(device)

    # 执行一些操作，比如加法，来测试GPU是否正常工作
    z = x + y
    print(z)
else:
    print("xpu is not available. Test on CPU.")