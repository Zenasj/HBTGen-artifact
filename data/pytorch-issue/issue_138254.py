import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch.utils.benchmark as benchmark
from functools import partial
from torchao.quantization import quantize_, int8_weight_only

def get_example_inputs():
    example_inputs = {
        "hidden_states": torch.randn(1, 4096, 64, dtype=torch.bfloat16, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="cuda"),
        "pooled_projections": torch.randn(1, 768, dtype=torch.bfloat16, device="cuda"),
        "timestep": torch.tensor([1.0], device="cuda"),
        "img_ids": torch.randn(4096, 3, dtype=torch.bfloat16, device="cuda"),
        "txt_ids": torch.randn(512, 3, dtype=torch.bfloat16, device="cuda"),
        "guidance": None,
        "joint_attention_kwargs": None,
        "return_dict": False
    }
    return example_inputs

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

def load_model():
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    return model

def aot_compile(name, model, **sample_kwargs):
    path = f"./{name}.so"
    print(f"{path=}")
    options = {
        "aot_inductor.output_path": path,
        "max_autotune": True,
        "triton.cudagraphs": True,
    }

    torch._inductor.aoti_compile_and_package(
        torch.export.export(model, (), sample_kwargs),
        (),
        sample_kwargs,
    )
    # torch._export.aot_compile(
    #     fn,
    #     (),
    #     sample_kwargs,
    #     options=options,
    #     disable_constraint_solver=True,
    # )
    return path

def aot_load(path):
    return torch._export.aot_load(path, "cuda")

@torch.no_grad()
def f(model, **kwargs):
    return model(**kwargs)

model = load_model()
quantize_(model, int8_weight_only())
inputs1 = get_example_inputs()
from torchao.utils import unwrap_tensor_subclass
unwrap_tensor_subclass(model)
path1 = aot_compile("bs_1_1024", model, **inputs1)

compiled_func_1 = aot_load(path1)
print(f"{compiled_func_1(**inputs1)[0].shape=}")

for _ in range(5):
    _ = compiled_func_1(**inputs1)[0]

time = benchmark_fn(f, compiled_func_1, **inputs1)
print(time)