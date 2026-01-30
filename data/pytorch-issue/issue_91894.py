import torch
torch.use_deterministic_algorithms(True)

frame_dim = 140
num_steps = 10
num_envs = 512


def go():
    torch.manual_seed(0)
    src = torch.randn(num_envs, num_steps, frame_dim).cuda()
    buffer = src.clone()

    # The problematic line - shift frames [0, 8] into [1, 9]
    buffer[:, 1 : num_steps] = buffer[:, 0 : (num_steps - 1)]

    print("Sum:", buffer.cpu().sum().item(), src.cpu().sum().item())


for i in range(10):
    go()

Sum: -413.53521728515625 -993.8636474609375
Sum: -132.9031982421875 -993.8636474609375
Sum: -186.0804443359375 -993.8636474609375
Sum: -177.52325439453125 -993.8636474609375
Sum: -24.764511108398438 -993.8636474609375
Sum: -245.3650665283203 -993.8636474609375
Sum: -408.0976867675781 -993.8636474609375
Sum: -464.16082763671875 -993.8636474609375
Sum: -417.3087463378906 -993.8636474609375