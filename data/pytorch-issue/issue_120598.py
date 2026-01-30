import torch
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

@torch.compile(backend='inductor')
def test(x):
    with torch.cuda.stream(stream1):
        x = x + 2.0
    stream2.wait_stream(stream1)
    with torch.cuda.stream(stream2):
        x = x + 3.0
    return x

input = torch.ones(3, 3, device='cuda')
output = test(input)
print('output = ', output.cpu())