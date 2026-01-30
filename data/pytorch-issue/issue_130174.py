from vllm import LLM

# Example prompts.
prompts = ["Hello, my name is", "The capital of France is"]
# Create an LLM with HF model name.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts.
outputs = llm.generate(prompts)