from vllm import LLM, SamplingParams

if __name__ == '__main__':
    # Sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM object
    llm = LLM(model="/data2/Llama-2-70b-hf", dtype="float16", tensor_parallel_size=4, enforce_eager=True, trust_remote_code=True)

    # Generate texts from the prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")