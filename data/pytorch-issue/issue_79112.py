import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

MODEL_CLASSES = {
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-large": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}


def main():
    model_type = "gpt2"
    model_class, tokenizer_class = MODEL_CLASSES[model_type]

    prompt_text = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered."""
    tokenizer = tokenizer_class.from_pretrained(model_type)
    model = model_class.from_pretrained(model_type)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    model.eval()
    device = torch.device("mps")
    model = model.to(device)
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, num_beams=2, max_length=500, num_return_sequences=2,
                             repetition_penalty=1.2, length_penalty=1.2, no_repeat_ngram_size=5, top_p=1.0,
                             early_stopping=True)
    ret = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for item in ret:
        print(item)


if __name__ == "__main__":
    main()