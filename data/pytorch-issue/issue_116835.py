from transformers import AutoTokenizer, GPTJForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs, labels=inputs["input_ids"])
input_ids = inputs["input_ids"]
out= model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)

import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.debugging import ExplainWithBackend

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        # <implement your compiler here>
        print("AOTAutograd produced a fx Graph in Aten IR:")
        # gm.print_readable()
        # gm.graph.print_tabular()
        # print(len(sample_inputs))
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=my_compiler
    )

torch._dynamo.reset()
fn = torch.compile(backend=toy_backend, dynamic=True,fullgraph=True)(model.generate)
out = fn(input_ids, do_sample=True, temperature=0.9, max_length=200)