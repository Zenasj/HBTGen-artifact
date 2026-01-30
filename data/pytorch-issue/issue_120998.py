import torch
@torch.compile()
def call(data):
	return data.view(torch.int32) + 1

data = torch.randint(0, 2**4, [4096, 4096], device='cuda', dtype=torch.int32)
out = call(data) #OK 
out = call(data.view(torch.float32)) #OK

import torch


@torch.compile()
def call(data):
    return data.view(torch.uint8) + 1


@torch.compile()
def print_tensor(data):
    print(data)


generator = torch.Generator(device="cuda").manual_seed(1234567890)

data = torch.randint(
    0, 2**4, [4096, 4096], generator=generator, device="cuda", dtype=torch.uint8
)
print("unmodified random tensor (uint8):")
print(data)

out = call(data)  # OK
print("random tensor + 1 (uint8):")
print(out)

out = call(data.view(torch.float16))  # was error
print("random tensor cast to float16, then back to uint8, then + 1:")
print(out)

print("random tensor cast to float16:")
print_tensor(data.view(torch.float16))

@torch.compile()
def call(data):
    return data.view(torch.uint8) + 1

    ...

    out = call(data.view(torch.float16))  # was error
    print("random tensor cast to float16, then back to uint8, then + 1:")
    print(out)

import torch
@torch.compile()
def call(data):
	return data.view(torch.uint8) + 1

data = torch.randint(0, 2**4, [4096, 4096], device='cuda', dtype=torch.uint8)
out = call(data) #OK 
out = call(data.view(torch.float16)) #Error

def compilable_cast(u8, dtype):
    n = dtype.itemsize
    bytes = [u8[..., i::n].to(dtype) for i in range(n)]
    if not LITTLE_BYTE_ORDER:
        bytes = bytes[::-1]

    bytes = sum(bytes[i] << (i * 8) for i in range(n))
    return bytes.view(dtype)