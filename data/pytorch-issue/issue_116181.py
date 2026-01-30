import torch

from transformers import AutoTokenizer, GPTJForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
input_ids = inputs["input_ids"]

start_time = time.time()
print("cpu.................")
fn_cpu = torch.compile(model)
with torch.no_grad():
    output_cpu = fn_cpu.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)
print(output_cpu)
print("--- %s pytorch with compile CPU seconds ---" % (time.time() - start_time))
torch._dynamo.reset()

from transformers import AutoTokenizer, GPTJForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
input_ids = inputs["input_ids"]

start_time = time.time()
print("cpu.................")
fn_cpu = torch.compile(model)
with torch.no_grad():
    output_cpu = fn_cpu(**inputs)
print(output_cpu)
print("--- %s pytorch with compile CPU seconds ---" % (time.time() - start_time))
torch._dynamo.reset()