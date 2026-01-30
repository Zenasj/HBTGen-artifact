import torch
import torch._dynamo

providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
            "do_copy_in_default_stream": True,
        },
    ),
    "CPUExecutionProvider",
]


@torch.compile(backend="onnxrt", options={"providers": providers})
def foo(x, y):
    return (x + y) * x


if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    print(foo(a, b))