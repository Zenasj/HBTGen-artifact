import torch


def f1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) / 2.0).round().int()


def f2(x: int, y: int) -> int:
    return int(round((x - y) / 2.0))


def main(enabled):
    print(f"Using torch version={torch.__version__} and optimization {enabled=}")
    example_inputs = (21, 16)
    for device in ["cpu", "cuda"]:
        out = f2(*example_inputs)

        example_inputs_tensor = tuple(torch.tensor(x).to(device) for x in example_inputs)
        out_torch = f1(*example_inputs_tensor)
        f1_traced = torch.jit.trace(f1, example_inputs=example_inputs_tensor)

        torch._C._jit_set_texpr_fuser_enabled(enabled)
        out_torch_traced = f1_traced(*example_inputs_tensor)

        print(f"[{device:4s}] Python = {out}, Torch = {out_torch}, Torch traced = {out_torch_traced}")


if __name__ == "__main__":
    for flag in True, False:
        main(enabled=flag)