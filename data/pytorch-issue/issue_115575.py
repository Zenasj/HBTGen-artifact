from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch._dynamo
from torch._inductor.compile_fx import compile_fx
import time
import types

torch.backends.cuda.matmul.allow_tf32 = True

model_name = "google/flan-t5-xl"
revision = "8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5"
max_new_tokens = 100

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device('cuda:0')

model_inputs = tokenizer('I have a weird performance regression', return_tensors='pt')
model_inputs.to(device)

with device:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16, revision=revision
    ).requires_grad_(False).eval()
    model.to(device)


def evaluate():
    t0 = time.time()
    output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens)
    t_elap = time.time()-t0
    print("Time per generated token: %.2f ms" % (t_elap/max_new_tokens*1000.0))


print("Before compile:")
evaluate()

# replace forward with compiled forward
compiled_forward = torch._dynamo.optimize(
    compile_fx,
    dynamic=True,
)(model.forward)
def override_forward_with_compile(self, *args, **kwargs):
    return compiled_forward(*args, **kwargs)
model.forward = types.MethodType(override_forward_with_compile, model)

# trigger compilation
output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens)

print("After compile:")
evaluate()

# replace forward with run forward
run_forward = torch._dynamo.run(compiled_forward)
def override_forward_with_run(self, *args, **kwargs):
    return run_forward(*args, **kwargs)
model.forward = types.MethodType(override_forward_with_run, model)

print("After run:")
evaluate()

torch/utils/collect_env.py