import torch.nn as nn

py
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for interpolate operator."""


class InterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_size, output_size, channels_last=False, mode='linear', dtype=torch.float):

        input_image = torch.randint(0, 256, size=input_size, dtype=dtype, device='cpu',
                                    requires_grad=self.auto_set())
        if channels_last:
            if input_image.ndim == 4:
                input_image = input_image.contiguous(memory_format=torch.channels_last)
            elif input_image.ndim == 5:
                input_image = input_image.contiguous(memory_format=torch.channels_last_3d)
            else:
                raise ValueError(
                    f"Can not set channels_last to the input of {input_image.ndim} dims"
                )


        align_corners = None if "nearest" in mode else False

        if mode == "linear":
            mode = {
                3: 'linear',
                4: 'bilinear',
                5: 'trilinear',
            }[input_image.ndim]

        self.inputs = {
            "input_image": input_image,
            "output_size": output_size,
            "mode": mode,
            "align_corners": align_corners,
        }

        self.set_module_name("interpolate")

    def forward(self, input_image, output_size, mode, align_corners):
        return torch.nn.functional.interpolate(input_image, size=output_size, mode=mode,
                                               align_corners=align_corners)


def make_config():
    sizes = (
        ((16, 320, 320), (8, 256, 256)),
        ((16, 320, 320), (32, 512, 512)),
    )

    attrs = []
    for (DHW1, DHW2) in sizes:
        attrs.append([(1, 3, *DHW1), DHW2])
        attrs.append([(1, 3, *DHW2), DHW1])


    config = op_bench.config_list(
        attr_names=["input_size", "output_size"],
        attrs=attrs,
        cross_product_configs={
            'channels_last': [True],
            'mode': ["linear", "nearest", "nearest-exact"],
            'dtype': [torch.float, torch.uint8]
        },
        tags=["short"],
    )

    # Need to remove instances with both torch.int and linear
    # Note: this is naaaasty
    def get_mode(l):
        for d in l:
            if "mode" in d:
                return d["mode"]
    def get_dtype(l):
        for d in l:
            if "dtype" in d:
                return d["dtype"]
    config = [l for l in config if not(get_mode(l) == "linear" and get_dtype(l) == torch.uint8)]
    return config

config = make_config()
op_bench.generate_pt_test(config, InterpolateBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

py
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("f3", nargs="?", default="main")
parser.add_argument("f2", nargs="?", default="new")
args = parser.parse_args()

with open(args.f1) as f:
    main = f.readlines()
with open(args.f2) as f:
    new = f.readlines()

out = []

for main_line, new_line in zip(main, new):
    # num_threads=1  # TODO: remove
    if main_line.startswith("num_threads="):
        num_threads = int(main_line.split("=")[-1])
    if main_line.startswith("# Input"):
        deets = f"{main_line.strip()}, {num_threads=}"
    if main_line.startswith("Forward"):
        main_time = float(main_line.split()[-1])
        new_time = float(new_line.split()[-1])
        ratio = main_time / new_time
        fmt = ".1f" if ratio < 3 else ".0f"
        improv = f"{ratio:{fmt}}X"
        time_fmt = ",.3f" if new_time < 100 else ",.1f"
        deets = deets.strip().replace("# Input: ", "")
        deets = deets.replace(": ", "=")
        deets = deets.replace("input_size=", "")
        deets = deets.replace(", output_size=", " -> ")
        deets = deets.replace("dtype=torch.", "")
        deets = deets.replace("mode=", "")
        deets = deets.replace("channels_last=True, ", "")
        split = deets.split(",")
        size = ','.join(split[:-3])
        mode, dtype, threads = split[-3:]
        deets = f"{size:<30} {mode:<15} {dtype:<10} {threads:<15}"

        l = f"{deets}  {improv:<5} {main_time / 1000:{time_fmt}}ms vs {new_time / 1000:{time_fmt}}ms"
        out.append(l)


def key(s):
    # s = ''.join(s.split()[1:]) # remove "N.nX" part
    num_threads = (int(re.findall(r"num_threads=(\d+)", s)[0]),)

    input_shape, output_shape = re.findall("\(.*?\)", s)
    input_shape = input_shape[1:-1]  # remove parenthesis
    input_HW = tuple(int(x) for x in input_shape.split(",")[-2:])
    input_C = (-int(input_shape.split(",")[1]),)

    output_HW = tuple(int(x) for x in output_shape[1:-1].split(","))
    is_downsample = (output_HW[0] < input_HW[0],)
    if "linear" in s:
        mode = "linear"
    elif "nearest-exact" in s:
        mode = "nearest-exact"
    else:
        assert "nearest" in s
        mode = "nearest"
    mode = (mode,)
    return is_downsample + input_HW + output_HW + num_threads + input_C + mode

for i, l in enumerate(sorted(out, key=key)):
    if i % 5 == 0:
        print()
    # if i % 10 == 0 and i % 40 != 0:
    #     print()
    # if i % 40 == 0:
    #     print("-" * 100)
    print(l)