import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.export import Dim


def get_example_inputs(prompt: str, tokenizer: AutoTokenizer) -> torch.tensor:
    """
    These arbitrary example inputs were observed by adding a debugger (`import pdb; pdb.set_trace()`)
    to the line corresponding this https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1197
    inside local venv site-packages.
    """
    example_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")
    seq_len = example_inputs["input_ids"].shape[1]
    example_inputs["position_ids"] = torch.arange(seq_len).unsqueeze(0).to("cuda")
    example_inputs["past_key_values"] = None
    example_inputs["inputs_embeds"] = None
    example_inputs["use_cache"] = False
    example_inputs["output_attentions"] = False
    example_inputs["output_hidden_states"] = False
    example_inputs["return_dict"] = True
    example_inputs["cache_position"] = torch.arange(seq_len).to("cuda")
    return example_inputs

@torch.no_grad()
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def aot_compile(path, model, **sample_kwargs):
    """
    torch.export.export + torch._inductor.aoti_compile_and_package the model,
    using given sample inputs.
    """
    seq_len_dim = Dim("seq_len", min=1, max=128)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "attention_mask": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
        "past_key_values": None,
        "inputs_embeds": None,
        "use_cache": None,
        "output_attentions": None,
        "output_hidden_states": None,
        "return_dict": None,
        "cache_position": {1: Dim("cache_position", min=1, max=128)},
    }
    exported_program = torch.export.export(
        model.model,
        (),
        sample_kwargs,
        dynamic_shapes=dynamic_shapes
    )

    return torch._inductor.aoti_compile_and_package(
        exported_program,
        (),
        sample_kwargs,
        package_path=path,
    )

def aot_load(path):
    return torch._inductor.aoti_load_package(path)

if __name__ == "__main__":
    model, tokenizer = load_model()

    prompt = "What is a compiler?"

    inputs1 = get_example_inputs(prompt, tokenizer)
    compile_path = aot_compile('llama3.pt2', model, **inputs1)
    print(f"AoT compiled path {compile_path}")

def get_example_inputs(prompt: str, tokenizer: AutoTokenizer) -> torch.tensor:
    example_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")
    return example_inputs

dynamic_shapes = {
        "input_ids": (1, seq_len_dim),
        "attention_mask": (1, seq_len_dim),
     }