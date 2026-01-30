import os
import torch
from vllm import LLM, SamplingParams
from vllm.plugins import set_torch_compile_backend

# make sure these models can be captured in full graph mode
os.environ["VLLM_TEST_DYNAMO_GRAPH_CAPTURE"] = "1"
os.environ["VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] = "1"

set_torch_compile_backend("inductor")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0)
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
          enforce_eager=True,
          dtype=torch.float16,
          quantization="fp8")
llm.generate(prompts, sampling_params)