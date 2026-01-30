import os
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import time
import torch
import torch.profiler
import oneccl_bindings_for_pytorch
from pyinstrument import Profiler

import torch
import os
import torch.backends.mkldnn
import torch.backends.openmp

print(f"Using {torch.get_num_threads()} threads (PyTorch)")
print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

# # Ensure PyTorch respects the OMP setting
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

# print(f"Now using {torch.get_num_threads()} threads after setting manually")



model_id = "meta-llama/Llama-3.1-8B-Instruct"

def main(is_tp, rank, world_size) -> None:
    backend = "ccl"
    print(is_tp)
    if is_tp:
        dist.init_process_group(backend)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if is_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    # Retrieve tensor parallel model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    print(model.dtype)

    # Prepare input tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Can I help" * 200
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512).input_ids.to(model.device)
    print(f"inpu shape is {inputs.shape}")

    # model = torch.compile(model)
    # warm-up
    if is_tp:
        dist.barrier()
    for i in range(5):
        with torch.no_grad():
            outputs = model(inputs)

    if is_tp:
        dist.barrier()
    # profiler = Profiler()
    # profiler.start()

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    # ) as prof:
    for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # profiler.stop()
    # with open(f"profile_tp_{is_tp}_backend_{backend}_rank_{rank}.html", "w") as f:
    #     f.write(profiler.output_html())

    count = 0
    for name, parameter in model.named_parameters():
        if isinstance(parameter.data, torch.distributed.tensor.DTensor):
            print(f"name: {name}\nparameter: {parameter}")
            original_shape = parameter.data.shape
            shape = parameter.data.to_local().shape
            print(f"paramater local shape is {shape}")
            print(f"paramater original shape is {original_shape}")
            count += 1
            if count > 2:
                break

    print(outputs)


if __name__ == "__main__":
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_tp = "RANK" in os.environ
    main(is_tp, rank, world_size)

import os
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import time
import torch
import torch.profiler
# import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
from pyinstrument import Profiler

import torch
import os
import torch.backends.mkldnn
import torch.backends.openmp

print(f"Using {torch.get_num_threads()} threads (PyTorch)")
print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

# # Ensure PyTorch respects the OMP setting
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

# print(f"Now using {torch.get_num_threads()} threads after setting manually")


model_id = "meta-llama/Llama-3.1-8B-Instruct"

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

def main(is_tp, rank, world_size) -> None:
    #backend = "ccl"
    print("is_tp, rank, world_size: ", is_tp, rank, world_size)
    #if is_tp:
    #    dist.init_process_group(backend)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if is_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    # Retrieve tensor parallel model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    print(model.dtype)
    # print(f"torch.max_memory_allocated() is {torch.max_memory_allocated()}")
    print("="*200)
    if dist.is_initialized():
        print("Backend:", dist.get_backend())
    else:
        print("Distributed process group is not initialized.")

    # Prepare input tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Can I help" * 200
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512).input_ids.to(model.device)
    print(f"inpu shape is {inputs.shape}")

    # model = torch.compile(model)
    # warm-up
    if is_tp:
        dist.barrier()
    for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")

    next_token_logits = outputs[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    response = tokenizer.decode(next_token)
    print(response)

    if is_tp:
        dist.barrier()
    # profiler = Profiler()
    # profiler.start()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
     for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model(inputs)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # profiler.stop()
    # with open(f"profile_tp_{is_tp}_backend_{backend}_rank_{rank}.html", "w") as f:
    #     f.write(profiler.output_html())

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        # for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=128, min_new_tokens=128)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    count = 0
    for name, parameter in model.named_parameters():
        if isinstance(parameter.data, torch.distributed.tensor.DTensor):
            print(f"name: {name}\nparameter: {parameter}")
            original_shape = parameter.data.shape
            shape = parameter.data.to_local().shape
            print(f"paramater local shape is {shape}")
            print(f"paramater original shape is {original_shape}")
            count += 1
            if count > 1:
                break

    print(outputs)


if __name__ == "__main__":
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_tp = world_size > 1
    main(is_tp, rank, world_size)

import os
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import time
import torch
import torch.profiler
# import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
from pyinstrument import Profiler

import torch
import os
import torch.backends.mkldnn
import torch.backends.openmp

print(f"Using {torch.get_num_threads()} threads (PyTorch)")
print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

# # Ensure PyTorch respects the OMP setting
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

# print(f"Now using {torch.get_num_threads()} threads after setting manually")


model_id = "meta-llama/Llama-3.1-8B-Instruct"

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

def main(is_tp, rank, world_size) -> None:
    #backend = "ccl"
    print("is_tp, rank, world_size: ", is_tp, rank, world_size)
    #if is_tp:
    #    dist.init_process_group(backend)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if is_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    # Retrieve tensor parallel model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    print(model.dtype)
    # print(f"torch.max_memory_allocated() is {torch.max_memory_allocated()}")
    print("="*200)
    if dist.is_initialized():
        print("Backend:", dist.get_backend())
    else:
        print("Distributed process group is not initialized.")

    # Prepare input tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "Can I help" * 200
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512).input_ids.to(model.device)
    print(f"inpu shape is {inputs.shape}")

    model.generation_config.cache_implementation = "static"
    if is_tp:
        model.config.hidden_size = model.config.hidden_size // 2
        model.config.num_key_value_heads = model.config.num_key_value_heads // 2

    # import pdb; pdb.set_trace()

    for i in range(1):
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=128, min_new_tokens=128)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")

    model.forward = torch.compile(model.forward)
    # warm-up
    if is_tp:
        dist.barrier()

    for i in range(4):
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=128, min_new_tokens=128)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        # for i in range(5):
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=128, min_new_tokens=128)
            end = time.time()
            print(f"time cost {(end-start)*1000} ms")

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    count = 0
    for name, parameter in model.named_parameters():
        if isinstance(parameter.data, torch.distributed.tensor.DTensor):
            print(f"name: {name}\nparameter: {parameter}")
            original_shape = parameter.data.shape
            shape = parameter.data.to_local().shape
            print(f"paramater local shape is {shape}")
            print(f"paramater original shape is {original_shape}")
            count += 1
            if count > 1:
                break

    print(outputs)


if __name__ == "__main__":
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_tp = world_size > 1
    main(is_tp, rank, world_size)