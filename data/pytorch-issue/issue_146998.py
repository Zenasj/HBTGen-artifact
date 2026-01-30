import torch.nn as nn

import torch
import time

configs = [
	[128, 128, 3, 1],
	[256, 256, 3, 1],
	[512, 512, 3, 1],
	[128, 256, 1, 1],
	[512, 512, 3, (2, 2, 2)],
	[256, 256, 3, (2, 2, 2)],
	[128, 3, 3, 1]
]

inputs = [
	[1, 128, 67, 258, 258],
	[1, 256, 35, 130, 130],
	[1, 512, 35, 130, 130],
	[1, 128, 67, 258, 258],
	[1, 512, 35, 130, 130],
	[1, 256, 27, 258, 258],
	[1, 128, 67, 258, 258],
]


def conv3dbenchmark(configs: list[list[int]], inputs: list[list[int]], repeat: int, dtype: torch.dtype, device: torch.device):
	modules = list()
	assert len(inputs) == len(configs)

	for config in configs:
		modules.append(torch.nn.Conv3d(config[0], config[1], config[2], stride=config[3]).to(device, dtype))

	for i in range(len(modules)):
		x = torch.randn(inputs[i]).to(device, dtype)
		print(f"Running Conv3d config: {configs[i]} input: {inputs[i]} type: {dtype}")
		start = time.perf_counter()
		for n in range(repeat):
			modules[i].forward(x)
		torch.cuda.synchronize(device)
		print(f"Time {(time.perf_counter() - start) / repeat} seconds\n")


if __name__ == "__main__":
	device = torch.device(0)

	conv3dbenchmark(configs, inputs, 5, torch.bfloat16, device)
	conv3dbenchmark(configs, inputs, 5, torch.float16, device)

torch.backends.cudnn.benchmark=true